"""Train the wav2vec2 -> AASIST spoof detector.

    python -m training.train --config training/config_train.yaml
    python -m training.train --config training/config_train.yaml --limit 400   # smoke

Two modes, chosen automatically (override with --mode):
  * frozen  — SSL features cached to disk once, AASIST head trains off the cache. Fast,
    CPU-viable. No waveform augmentation. ~5-10% EER.
  * e2e     — waveform -> (RawBoost + MUSAN/RIR environmental + telephone channel) ->
    trainable SSL frontend -> AASIST, end to end. Needs a GPU. Frontend frozen for
    `stage2_unfreeze_epoch` epochs then unfrozen at lr*mult. This is the quality path
    (~<1% EER on ASVspoof).

Best dev-EER checkpoint -> backend/models/aasist_indicw2v.pt (the live pipeline auto-loads
it; if the frontend was fine-tuned, its weights ride along in the same file).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


def _device(cfg: dict) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


def _balanced_head(samples, n_total):
    """Deterministic balanced subset: ~n_total/2 per class from the shuffled pool.

    Raises if either class is entirely absent from ``samples``. Silently returning a
    single-class subset here means every epoch's dev EER gets an empty score list for the
    missing class, ``compute_eer`` returns ``nan``, and ``nan < best`` is always False --
    training finishes with "best dev EER inf%" and no traceback at all, having never once
    saved a checkpoint. Fail loudly instead, right where the actual problem is (the source
    manifest), not several silent steps downstream.
    """
    import random

    rng = random.Random(0)
    per = max(n_total // 2, 1)
    out = []
    for lab in (0, 1):
        pool = [s for s in samples if s.label == lab]
        if not pool:
            raise ValueError(
                f"_balanced_head: no samples with label={lab} among {len(samples)} rows -- "
                "can't build a balanced subset. The source manifest is missing a class "
                "(check prepare_manifests.py output / which datasets got attached), not a "
                "subsampling bug."
            )
        rng.shuffle(pool)
        out += pool[:per]
    rng.shuffle(out)
    return out


def _capped_manifest(path: str, per_class: int) -> str:
    """Write a per-class-capped copy of a manifest next to it; return the new path."""
    import csv
    import random

    src = ROOT / path
    rows = list(csv.DictReader(src.open(encoding="utf-8"), delimiter="\t"))
    by_lab: dict[str, list] = {}
    for r in rows:
        by_lab.setdefault(r["label"], []).append(r)
    rng = random.Random(0)
    kept = []
    for lab, items in by_lab.items():
        rng.shuffle(items)
        kept += items[:per_class]
    dst = src.with_name(src.stem + f".cap{per_class}.tsv")
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(kept)
    print(f"capped {src.name} -> {dst.name}: {len(kept)} rows ({per_class}/class)")
    return str(dst.relative_to(ROOT))


def _pick_mode(cfg: dict, device: torch.device, override: str | None) -> str:
    if override:
        return override
    fe = cfg.get("frontend", {})
    if device.type == "cuda" or fe.get("stage2_unfreeze_epoch") is not None:
        return "e2e"
    return "frozen"


# ----------------------------------------------------------------- scoring
@torch.no_grad()
def _eer_over_loader(score_batch, loader, device) -> tuple[float, int]:
    from training.metrics import compute_eer

    bona, spoof = [], []
    for batch in loader:
        x, labels = batch[0].to(device), batch[1]
        for s, y in zip(score_batch(x), labels.numpy()):
            (spoof if y == 1 else bona).append(float(s))
    eer, _ = compute_eer(np.array(bona), np.array(spoof))
    return eer, len(bona) + len(spoof)


def _spoof_scorer(model, loss_fn):
    uses_emb = getattr(loss_fn, "uses_embedding", False)

    def score(x):
        if uses_emb:
            _, emb = model(x, return_embedding=True)
            return (-loss_fn.score(emb)).cpu().numpy()
        return torch.softmax(model(x), dim=-1)[:, 1].cpu().numpy()

    return score


# ----------------------------------------------------------------- frozen mode (cached feats)
def _run_frozen(cfg, device, args) -> float:
    from backend.inference.classifier import AASISTClassifier
    from training.dataset import FeatureDataset, collate_features
    from training.features import build_cache
    from training.losses import make_loss

    # cap the (huge, imbalanced) ASVspoof train set for the frozen baseline so the feature
    # cache stays a sane size — the e2e/GPU path uses all of it from waveforms.
    cap = None if args.limit else cfg.get("frozen_max_per_class")
    train_manifest = _capped_manifest(cfg["manifests"]["train"], cap) if cap else cfg["manifests"]["train"]

    cache = build_cache(train_manifest, cfg, limit=args.limit)
    build_cache(cfg["manifests"]["dev"], cfg, limit=(args.limit // 4 if args.limit else None))

    mf = int(cfg.get("max_frames", 400))
    dev_lim = max(args.limit // 4, 8) if args.limit else cfg.get("dev_subsample")
    tr = FeatureDataset(train_manifest, cache, max_frames=mf, train=True, limit=args.limit)
    dv = FeatureDataset(cfg["manifests"]["dev"], cache, max_frames=mf, train=False)
    if dev_lim:
        dv.samples = _balanced_head(dv.samples, dev_lim)
    bs = int(cfg.get("batch_size", 32))
    tr_dl = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate_features,
                       num_workers=int(cfg.get("num_workers", 0)))
    dv_dl = DataLoader(dv, batch_size=bs, shuffle=False, collate_fn=collate_features)

    feat_dim = int(np.load(next(cache.root.glob("*.npy"))).shape[1])
    model = AASISTClassifier(feat_dim=feat_dim, embed_dim=cfg["classifier"]["embed_dim"],
                             num_classes=cfg["classifier"]["num_classes"]).to(device)
    loss_fn = make_loss(cfg.get("loss", {}), feat_dim=cfg["classifier"]["embed_dim"]).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()),
                           lr=cfg.get("optimizer", {}).get("lr", 1e-4),
                           weight_decay=cfg.get("optimizer", {}).get("weight_decay", 1e-4))
    print(f"[frozen] train={len(tr)} dev={len(dv)} feat_dim={feat_dim}")

    scorer = _spoof_scorer(model, loss_fn)
    return _train_loop(cfg, args, model, loss_fn, opt, tr_dl, dv_dl, device, scorer,
                       save_state=lambda: {
                           "model": model.state_dict(),
                           "loss_state": loss_fn.state_dict(),
                           "feat_dim": feat_dim, "config": cfg,
                           "embed_dim": cfg["classifier"]["embed_dim"],
                           "num_classes": cfg["classifier"]["num_classes"],
                           "frontend_model_id": cfg["frontend"]["model_id"],
                           "layer": cfg["frontend"]["layer"],
                       })


def _compose_augment(*fns):
    """Chain waveform augmenters into one callable(np.ndarray, label)->np.ndarray for
    ManifestDataset's single ``augment`` slot. Each fn gates itself (own probability, optionally
    label-conditional — see TelephoneChannelAugment/EnvironmentalAugment's p_spoof), so a
    sample can draw any subset of them independently."""
    fns = [f for f in fns if f is not None]
    if not fns:
        return None

    def _apply(wav, label=None):
        for f in fns:
            wav = f(wav, label)
        return wav

    return _apply


# ----------------------------------------------------------------- e2e mode (waveform + RawBoost + environmental + telephone)
def _run_e2e(cfg, device, args) -> float:
    from training.dataset import ManifestDataset, collate_waveforms
    from training.losses import make_loss
    from training.model import EndToEndDetector

    fe = cfg["frontend"]
    aug = cfg.get("augment", {})
    rb = None
    if aug.get("rawboost"):
        from training.augment import RawBoost

        rb = RawBoost(mode=int(aug.get("rawboost_mode", 5)), p=float(aug.get("rawboost_p", 0.5)))

    env = None
    if aug.get("environmental"):
        from training.augment_environmental import EnvironmentalAugment

        env_p_spoof = aug.get("environmental_p_spoof")
        env = EnvironmentalAugment(
            musan_dir=aug.get("musan_dir"),
            rir_dir=aug.get("rir_dir"),
            mode=aug.get("environmental_mode", "both"),
            p=float(aug.get("environmental_p", 0.5)),
            p_spoof=float(env_p_spoof) if env_p_spoof is not None else None,
            snr_min_db=float(aug.get("environmental_snr_min_db", 0.0)),
            snr_max_db=float(aug.get("environmental_snr_max_db", 20.0)),
        )

    tel = None
    if aug.get("telephone"):
        from training.augment_telephone import TelephoneChannelAugment

        tel_p_spoof = aug.get("telephone_p_spoof")
        tel = TelephoneChannelAugment(
            low_hz=float(aug.get("telephone_low_hz", 300.0)),
            high_hz=float(aug.get("telephone_high_hz", 3400.0)),
            p=float(aug.get("telephone_p", 0.45)),
            p_spoof=float(tel_p_spoof) if tel_p_spoof is not None else None,
            snr_min_db=float(aug.get("telephone_snr_min_db", 15.0)),
            snr_max_db=float(aug.get("telephone_snr_max_db", 35.0)),
        )

    combined_aug = _compose_augment(rb, env, tel)

    crop = float(cfg.get("crop_seconds", 4.0))
    dev_lim = max(args.limit // 4, 8) if args.limit else cfg.get("dev_subsample")
    tr = ManifestDataset(cfg["manifests"]["train"], crop_seconds=crop, train=True,
                         augment=combined_aug, limit=args.limit)
    dv = ManifestDataset(cfg["manifests"]["dev"], crop_seconds=crop, train=False)
    if dev_lim:
        dv.samples = _balanced_head(dv.samples, dev_lim)
    bs = int(cfg.get("batch_size", 8))

    # Build the parquet Arrow index (per split) in THIS process first — forked DataLoader
    # workers then just mmap the finished file (a fresh build inside a worker can deadlock).
    from training.hf_audio import HFAudioStore, is_ref, parse_ref

    seen_splits = set()
    for s in list(tr.samples) + list(dv.samples):
        if is_ref(s.path):
            key = parse_ref(s.path)[:2]
            if key not in seen_splits:
                seen_splits.add(key)
                HFAudioStore.load(s.path)

    # ASVspoof train is ~9x spoof; sample classes evenly so bonafide isn't drowned out
    labels = np.array([s.label for s in tr.samples])
    w = np.where(labels == 1, 1.0 / max((labels == 1).sum(), 1),
                 1.0 / max((labels == 0).sum(), 1))
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                     num_samples=len(tr), replacement=True)
    nw = int(cfg.get("num_workers", 2))
    tr_dl = DataLoader(tr, batch_size=bs, sampler=sampler, collate_fn=collate_waveforms,
                       num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    dv_dl = DataLoader(dv, batch_size=bs, shuffle=False, collate_fn=collate_waveforms)

    model = EndToEndDetector(model_id=fe["model_id"], layer=fe["layer"],
                             embed_dim=cfg["classifier"]["embed_dim"],
                             num_classes=cfg["classifier"]["num_classes"],
                             frontend_trainable=False).to(device)
    loss_fn = make_loss(cfg.get("loss", {}), feat_dim=cfg["classifier"]["embed_dim"]).to(device)

    o = cfg.get("optimizer", {})
    head_params = list(model.classifier.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(head_params, lr=o.get("lr", 1e-4),
                            weight_decay=o.get("weight_decay", 1e-4))
    print(f"[e2e] train={len(tr)} dev={len(dv)} frontend={fe['model_id']} "
          f"rawboost={rb is not None} environmental={env is not None} telephone={tel is not None}")

    unfreeze_ep = fe.get("stage2_unfreeze_epoch")
    fe_lr = o.get("lr", 1e-4) * o.get("frontend_lr_mult", 0.1)

    def on_epoch_start(ep):
        nonlocal opt
        if unfreeze_ep is not None and ep >= unfreeze_ep + 1 and not model.frontend_trainable:
            model.set_frontend_trainable(True)
            opt = torch.optim.AdamW(
                [{"params": head_params, "lr": o.get("lr", 1e-4)},
                 {"params": model.frontend.parameters(), "lr": fe_lr}],
                weight_decay=o.get("weight_decay", 1e-4),
            )
            print(f"  epoch {ep}: unfroze frontend (lr={fe_lr:g})")

    scorer = _spoof_scorer(model, loss_fn)
    return _train_loop(cfg, args, model, loss_fn, opt, tr_dl, dv_dl, device, scorer,
                       save_state=model.state_for_checkpoint, on_epoch_start=on_epoch_start,
                       get_opt=lambda: opt)


# ----------------------------------------------------------------- shared loop
def _train_loop(cfg, args, model, loss_fn, opt, tr_dl, dv_dl, device, scorer,
                save_state, on_epoch_start=None, get_opt=None) -> float:
    epochs = args.epochs or int(cfg.get("epochs", 30))
    accum = int(cfg.get("grad_accum", 1))
    out = ROOT / cfg["checkpoint"]["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out.with_suffix(".resume.pt")
    uses_emb = getattr(loss_fn, "uses_embedding", False)
    best = float("inf")
    start_ep = 1

    if resume_path.exists() and not args.limit:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        start_ep = ck["epoch"] + 1
        best = ck["best"]
        if on_epoch_start:
            on_epoch_start(start_ep)  # re-apply unfreeze if we're past it
        model.load_state_dict(ck["model"])
        loss_fn.load_state_dict(ck["loss"])
        (get_opt() if get_opt else opt).load_state_dict(ck["opt"])
        print(f"resumed from {resume_path.name}: epoch {start_ep}, best EER {best*100:.2f}%")

    ws = getattr(args, "warm_start", None)
    if start_ep == 1 and ws and Path(ws).exists() and not args.limit:
        # no <out>.resume.pt (e.g. a fresh Kaggle session after the 12 h wall), but a serving
        # checkpoint from an earlier run is available — continue from its epoch with the
        # trained weights. Optimizer state and best-EER history are gone, so AdamW restarts
        # cold (rebuilds its moments in a few dozen steps) and the best-EER latch resets.
        w = torch.load(ws, map_location=device, weights_only=False)
        start_ep = int(w["epoch"]) + 1
        if on_epoch_start:
            on_epoch_start(start_ep)  # unfreeze frontend + rebuild opt before loading weights
        if hasattr(model, "classifier") and "model" in w:
            model.classifier.load_state_dict(w["model"])
            if w.get("frontend") is not None:
                model.frontend.load_state_dict(w["frontend"])
        else:
            model.load_state_dict(w["model"])
        if uses_emb and w.get("oc_softmax"):
            loss_fn.load_state_dict(w["oc_softmax"]["state"])
        print(f"warm-started from {Path(ws).name}: continuing at epoch {start_ep} "
              f"(fresh optimizer; best-EER tracking reset)", flush=True)

    for ep in range(start_ep, epochs + 1):
        if on_epoch_start:
            on_epoch_start(ep)
        optimizer = get_opt() if get_opt else opt
        model.train()
        if hasattr(model, "frontend") and not model.frontend_trainable:
            model.frontend.eval()
        t0, tot, n = time.time(), 0.0, 0
        n_batches = len(tr_dl)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tr_dl):
            x, y = batch[0].to(device), batch[1].to(device)
            if uses_emb:
                logits, emb = model(x, return_embedding=True)
                loss = loss_fn(emb, y)
            else:
                loss = loss_fn(model(x), y)
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tot += float(loss.detach()) * len(y)
            n += len(y)
            if i == 0 or (i + 1) % 200 == 0:
                print(f"  ep{ep} batch {i+1}/{n_batches}  loss={tot/max(n,1):.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        eer, n_dev = _eer_over_loader(scorer, dv_dl, device)
        flag = ""
        if eer < best:
            best = eer
            st = save_state()
            st["dev_eer"] = eer
            st["epoch"] = ep
            if uses_emb:  # OC-Softmax: the centre IS the model at inference time — save it
                st["oc_softmax"] = {
                    "state": {k: v.cpu() for k, v in loss_fn.state_dict().items()},
                    "m_real": getattr(loss_fn, "m_real", 0.9),
                    "m_fake": getattr(loss_fn, "m_fake", 0.2),
                    "alpha": getattr(loss_fn, "alpha", 20.0),
                }
            torch.save(st, out)
            flag = "  <- best, saved"
        # resume state every epoch (killed session -> re-run continues from here)
        if not args.limit:
            torch.save({"epoch": ep, "best": best, "model": model.state_dict(),
                        "loss": loss_fn.state_dict(),
                        "opt": (get_opt() if get_opt else opt).state_dict()}, resume_path)
        print(f"epoch {ep:2d}/{epochs}  loss={tot/max(n,1):.4f}  dev_EER={eer*100:.2f}%  "
              f"({n_dev} dev, {time.time()-t0:.0f}s){flag}", flush=True)

    print(f"\nbest dev EER {best*100:.2f}%  ->  {out}")
    resume_path.unlink(missing_ok=True)
    return best


# ----------------------------------------------------------------- keep the old helpers importable
def train_one_epoch(model, loss_fn, loader, opt, device) -> float:
    """Legacy single-epoch helper (cached-feature batches). Used by tests."""
    model.train()
    total, n = 0.0, 0
    for feats, labels, _ in loader:
        feats, labels = feats.to(device), labels.to(device)
        if getattr(loss_fn, "uses_embedding", False):
            _, emb = model(feats, return_embedding=True)
            loss = loss_fn(emb, labels)
        else:
            loss = loss_fn(model(feats), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(labels)
        n += len(labels)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_dev(model, loss_fn, loader, device) -> tuple[float, float]:
    from training.metrics import compute_eer

    model.eval()
    bona, spoof = [], []
    for feats, labels, _ in loader:
        feats = feats.to(device)
        if getattr(loss_fn, "uses_embedding", False):
            _, emb = model(feats, return_embedding=True)
            score = -loss_fn.score(emb).cpu().numpy()
        else:
            score = torch.softmax(model(feats), dim=-1)[:, 1].cpu().numpy()
        for s, y in zip(score, labels.numpy()):
            (spoof if y == 1 else bona).append(float(s))
    eer, _ = compute_eer(np.array(bona), np.array(spoof))
    return eer, len(bona) + len(spoof)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "training" / "config_train.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="ABSOLUTE target epoch to train through (overrides cfg epochs), NOT a "
                         "count of additional epochs to run. The loop is range(start_ep, "
                         "epochs+1) — with --resume/--warm-start, start_ep is already past 1, "
                         "so e.g. resuming at epoch 24 needs --epochs 24 (not --epochs 1) to "
                         "run that epoch; too low silently produces an empty range (zero "
                         "batches, no error, best dev EER stays inf%%)")
    ap.add_argument("--mode", choices=["frozen", "e2e"], default=None)
    ap.add_argument("--warm-start", type=Path, default=None,
                    help="continue from a serving checkpoint (model + OC-Softmax centre, no "
                         "optimizer state) when no <out>.resume.pt is present — used to chain "
                         "Kaggle runs across the 12 h wall")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(cfg.get("seed", 42))
    device = _device(cfg)
    mode = _pick_mode(cfg, device, args.mode)
    print(f"device={device}  mode={mode}")

    best = _run_e2e(cfg, device, args) if mode == "e2e" else _run_frozen(cfg, device, args)
    return 0 if best < 0.5 else 0  # non-fatal even if EER is poor on a smoke run


if __name__ == "__main__":
    raise SystemExit(main())
