"""Streaming responses module for LLM."""
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextIteratorStreamer,
    pipeline,
)


def stream_llm(
    llm_input: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
) -> Generator[str, None, None]:
    """Streams response from LLM."""
    stream = TextIteratorStreamer(tokenizer=tokenizer)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=stream,
        device_map="auto",
    )
    with ThreadPoolExecutor() as executor:
        executor.submit(pipe, llm_input, max_new_tokens=100)
        yield from stream
