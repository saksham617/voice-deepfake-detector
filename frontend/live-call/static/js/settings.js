// Phone-verification flow. Talks only to our backend (/twilio/*); Twilio creds stay server-side.
(() => {
  const $ = (id) => document.getElementById(id);
  const show = (id, on) => { $(id).hidden = !on; };
  const msg = (text, kind = "") => { const m = $("msg"); m.textContent = text; m.className = "msg " + kind; };

  async function api(path, body) {
    const res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    return data;
  }

  function renderProfile(p) {
    $("protectedNumber").textContent = p.protected_number || "— not configured —";
    if (p.dev_mode) { show("devHint", true); $("devCode").textContent = "123456"; }
    if (p.profile && p.profile.phone_verified) {
      show("step-phone", false); show("step-code", false); show("step-verified", true);
      $("verifiedNumber").textContent = p.profile.phone;
      $("statusRow").textContent = "Status: verified ✓";
    } else {
      $("statusRow").textContent = "Status: not verified yet";
    }
  }

  async function sendCode() {
    const phone = $("phone").value.trim();
    if (!phone) return msg("Enter your phone number first.", "err");
    $("sendBtn").disabled = true;
    try {
      const r = await api("/twilio/verify/start", { phone });
      msg(r.dev_mode ? "Dev-mock: enter the code below." : "Code sent — check your SMS.", "ok");
      show("step-phone", false); show("step-code", true);
    } catch (e) { msg(e.message, "err"); }
    finally { $("sendBtn").disabled = false; }
  }

  async function verify() {
    const phone = $("phone").value.trim();
    const code = $("code").value.trim();
    if (!code) return msg("Enter the code.", "err");
    $("verifyBtn").disabled = true;
    try {
      const r = await api("/twilio/verify/check", { phone, code });
      show("step-code", false); show("step-verified", true);
      $("verifiedNumber").textContent = r.profile.phone;
      $("statusRow").textContent = "Status: verified ✓";
      msg("Phone verified.", "ok");
    } catch (e) { msg(e.message, "err"); }
    finally { $("verifyBtn").disabled = false; }
  }

  function reset() {
    show("step-code", false); show("step-verified", false); show("step-phone", true);
    msg("");
  }

  $("sendBtn").addEventListener("click", sendCode);
  $("verifyBtn").addEventListener("click", verify);
  $("backBtn").addEventListener("click", reset);

  api("/twilio/profile").then(renderProfile).catch((e) => { $("statusRow").textContent = e.message; });
})();
