"""
memory_utils.py

Utility functions for ACMQ-RAG.

This module provides reusable helper functions used across
the memory construction pipeline.

Author: Vijender Reddy
Project: ACMQ-RAG
"""

import hashlib
import re
from datetime import datetime
from typing import List

from .memory_types import MemoryUnit


# ----------------------------------------------------------------------
# Text Utilities
# ----------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - lowercase
    - remove extra spaces
    - remove leading/trailing spaces
    """

    if text is None:
        return ""

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------------------------------------------
# Triple Utilities
# ----------------------------------------------------------------------

def normalize_memory(memory: MemoryUnit) -> MemoryUnit:
    """
    Normalize all fields of a MemoryUnit.
    """

    memory.subject = normalize_text(memory.subject)
    memory.relation = normalize_text(memory.relation)
    memory.object = normalize_text(memory.object)

    memory.normalized_text = (
        f"{memory.subject}|"
        f"{memory.relation}|"
        f"{memory.object}"
    )

    return memory


# ----------------------------------------------------------------------
# Hash Utilities
# ----------------------------------------------------------------------

def generate_memory_hash(memory: MemoryUnit) -> str:
    """
    Generate a unique hash for a MemoryUnit.
    """

    text = (
        normalize_text(memory.subject)
        + "|"
        + normalize_text(memory.relation)
        + "|"
        + normalize_text(memory.object)
    )

    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------
# Duplicate Detection
# ----------------------------------------------------------------------

def is_duplicate(
    memory: MemoryUnit,
    existing_memories: List[MemoryUnit]
) -> bool:
    """
    Check whether the memory already exists.
    """

    target_hash = generate_memory_hash(memory)

    for item in existing_memories:

        if generate_memory_hash(item) == target_hash:
            return True

    return False


# ----------------------------------------------------------------------
# Similarity
# ----------------------------------------------------------------------

def same_subject(memory1: MemoryUnit, memory2: MemoryUnit) -> bool:

    return (
        normalize_text(memory1.subject)
        ==
        normalize_text(memory2.subject)
    )


def same_relation(memory1: MemoryUnit, memory2: MemoryUnit) -> bool:

    return (
        normalize_text(memory1.relation)
        ==
        normalize_text(memory2.relation)
    )


def same_object(memory1: MemoryUnit, memory2: MemoryUnit) -> bool:

    return (
        normalize_text(memory1.object)
        ==
        normalize_text(memory2.object)
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def is_valid_memory(memory: MemoryUnit) -> bool:
    """
    Check whether a MemoryUnit is valid.
    """

    if memory.subject == "":
        return False

    if memory.relation == "":
        return False

    if memory.object == "":
        return False

    return True


# ----------------------------------------------------------------------
# Timestamp Utilities
# ----------------------------------------------------------------------

def current_timestamp() -> str:
    """
    Return current timestamp.
    """

    return datetime.utcnow().isoformat()


def days_between(date1: str, date2: str) -> int:
    """
    Calculate difference in days.
    """

    d1 = datetime.fromisoformat(date1)
    d2 = datetime.fromisoformat(date2)

    return abs((d2 - d1).days)


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------

def memory_to_string(memory: MemoryUnit) -> str:
    """
    Pretty string representation.
    """

    return (
        f"{memory.subject}"
        f" --[{memory.relation}]--> "
        f"{memory.object}"
    )


# ----------------------------------------------------------------------
# Batch Utilities
# ----------------------------------------------------------------------

def normalize_memory_list(
    memories: List[MemoryUnit]
) -> List[MemoryUnit]:

    return [normalize_memory(m) for m in memories]


def filter_invalid_memories(
    memories: List[MemoryUnit]
) -> List[MemoryUnit]:

    return [m for m in memories if is_valid_memory(m)]