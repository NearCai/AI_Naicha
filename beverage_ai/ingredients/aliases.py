"""Alias / synonym resolution for out-of-vocab ingredient names.

Corresponds to 技术方案书 §D.5 step 1.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .vocab import Vocab, _default_data_dir


class AliasResolver:
    def __init__(self, mapping: dict[str, str], vocab: Vocab | None = None):
        # Lowercase keys for case-insensitive lookup
        self._map = {k.strip().lower(): v for k, v in mapping.items()}
        self._vocab = vocab

    @classmethod
    def from_yaml(cls, path: str | Path, vocab: Vocab | None = None) -> AliasResolver:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw, vocab)

    def resolve(self, name: str) -> str | None:
        """Return canonical vocab id, or None if no alias known."""
        canonical = self._map.get(name.strip().lower())
        if canonical is None:
            return None
        if self._vocab is not None and canonical not in self._vocab:
            return None
        return canonical


def load_default_aliases(vocab: Vocab | None = None) -> AliasResolver:
    return AliasResolver.from_yaml(_default_data_dir() / "ingredients" / "aliases.yaml", vocab)
