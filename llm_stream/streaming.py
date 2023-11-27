"""Streaming responses module for LLM."""
import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Self

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextStreamer,
    pipeline,
)


async def stream_llm(
    llm_input: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> AsyncGenerator[str, None]:
    """Streams response from LLM."""
    stream = AsyncTextIteratorStreamer(tokenizer=tokenizer)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=stream,
        device_map="auto",
    )
    with ThreadPoolExecutor() as executor:
        executor.submit(pipe, llm_input, max_new_tokens=100)
        async for output in stream:
            yield output


class AsyncTextIteratorStreamer(TextStreamer):  # type: ignore[misc]
    """Test implementation of an asyncio-friendly verion of TextIteratorStreamer."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        *,
        skip_prompt: bool = False,
        **decode_kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.stop_signal = None
        self.loop = asyncio.get_running_loop()

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:  # noqa: FBT001, FBT002
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        asyncio.run_coroutine_threadsafe(
            self.text_queue.put(text),
            self.loop,
        )
        if stream_end:
            asyncio.run_coroutine_threadsafe(
                self.text_queue.put(self.stop_signal),
                self.loop,
            )

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> str:
        value = await self.text_queue.get()
        if value is self.stop_signal:
            raise StopAsyncIteration
        return value
