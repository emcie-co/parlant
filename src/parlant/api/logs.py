from fastapi import APIRouter, WebSocket

from parlant.adapters.loggers.websocket import WebSocketLogger


def create_router(
    websocket_logger: WebSocketLogger,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/logs")
    async def stream_logs(websocket: WebSocket) -> None:
        await websocket.accept()
        subscription = await websocket_logger.subscribe(websocket)
        await subscription.expiration.wait()

    return router
