import asyncio
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.core.logging import Logger


def create_router(
    websocket_logger: WebSocketLogger,
    logger: Logger,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/logs")
    async def stream_logs(websocket: WebSocket) -> Any:
        logger.info("WebSocket connection application recieved.")
        await websocket.accept()
        logger.info("WebSocket connection accepted.")

        logger_id = websocket_logger.append(websocket)
        logger.info(f"WebSocket logger registered with ID: {logger_id}")

        try:
            while True:
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
        finally:
            websocket_logger.remove(logger_id)
            logger.info(f"WebSocket logger with ID {logger_id} removed.")

    return router
