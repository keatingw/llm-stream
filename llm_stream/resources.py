"""Resource retrieval module."""
import os
from functools import cache

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)


@cache
def get_model(
    pretrained_model_name_or_path: str | os.PathLike[str]
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Retrieves model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        device_map="auto",
    )
    return model, tokenizer
