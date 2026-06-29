import importlib.metadata
from packaging import version
from functools import lru_cache
import torch

def is_flash_attn_2_available():
    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    else:
        return False

@lru_cache()
def is_flash_attn_greater_or_equal(library_version: str):
    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)

@lru_cache()
def is_flash_attn_greater_or_equal_2_10():
    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")