"""Module for API components."""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from llm_stream.resources import get_model
from llm_stream.streaming import stream_llm


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Setup for FastAPI app resources."""
    model, tokenizer = get_model("facebook/opt-125m")
    app.state.model = model
    app.state.tokenizer = tokenizer
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def llm_websocket(
    websocket: WebSocket,
) -> None:
    """LLM websocket handler."""
    await websocket.accept()
    while True:
        msg = await websocket.receive_text()
        for token in stream_llm(
            msg, websocket.app.state.model, websocket.app.state.tokenizer
        ):
            await websocket.send_text(token)
