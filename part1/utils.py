"""Deterministic, label-preserving-by-design spelling perturbations.

Semantic label preservation still requires human inspection of real examples.
This second version is an explicitly disclosed revision after the p30 test result.
The p60 rule is fixed before this new evaluation; original results are preserved.
"""

import hashlib
import random
import re

TRANSFORM_VERSION = "internal_transposition_v2_p60"
TRANSFORM_SEED = 0
TRANSFORM_PROBABILITY = 0.60
_PROTECTED_WORDS = {
    "no", "not", "never", "neither", "nor", "none", "nobody", "nothing",
    "nowhere", "without", "cannot", "hardly", "barely", "scarcely",
}
_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def example_transform(example):
    return {**example, "text": example["text"].lower()}


def custom_transform(example):
    """Swap one internal adjacent letter pair in 60% of eligible words.

    Keep punctuation, spacing, casing, first/last letters, contractions, short
    words and explicit negators. Use a text-derived local RNG so repeated input
    is identical across processes, batching and dataset traversal orders.
    Return a new mapping; never alter labels or the caller's original example.
    """
    text = example["text"]
    digest = hashlib.sha256(
        f"{TRANSFORM_VERSION}:{TRANSFORM_SEED}:{text}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))

    def transform_word(match):
        word = match.group(0)
        if (len(word) < 5 or word.lower() in _PROTECTED_WORDS
                or "'" in word or "’" in word):
            return word
        # Both characters must be internal and different to change the spelling.
        positions = [i for i in range(1, len(word) - 2)
                     if word[i].lower() != word[i + 1].lower()]
        if not positions or rng.random() >= TRANSFORM_PROBABILITY:
            return word
        i = rng.choice(positions)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    return {**example, "text": _WORD.sub(transform_word, text)}
