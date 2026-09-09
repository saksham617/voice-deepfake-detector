from .legacy import router as legacy_router
from .rest import router as rest_router
from .websocket import router as ws_router

__all__ = ["legacy_router", "rest_router", "ws_router"]
