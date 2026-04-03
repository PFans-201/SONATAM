"""
dataset.py
==========
PyTorch Dataset wrappers for the curated harmonic dataset.

Classes
-------
HarmonicDataset
    Wraps the curated DataFrame.  Each item is either:
      * a flat feature vector + genre label     (mode='classification')
      * a chord token sequence + genre label    (mode='sequence')

Functions
---------
build_vocab(df, col) → {token: int}
    Build a vocabulary dict mapping token strings to integer indices.

Dependencies
------------
    pip install torch scikit-learn
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["HarmonicDataset", "build_vocab"]

# ── Numeric feature columns used in 'classification' mode ────────────────
NUMERIC_FEATURES = [
    "major_mode_ratio",
    "chord_vocab_roman",
    "unique_chord_ratio",
    "func_ratio_T",
    "func_ratio_D",
    "func_ratio_S",
    "func_ratio_PD",
    "harm_rhythm_mean",
    "harm_rhythm_std",
    "avg_chord_cardinality",
    "dyad_ratio",
    "transition_entropy",
    "num_modulations",
    "modulations_per_measure",
    "msd_tempo",
    "msd_duration",
    "msd_loudness",
    "msd_danceability",
    "msd_energy",
]


def build_vocab(df: pd.DataFrame, col: str = "roman") -> Dict[str, int]:
    """
    Build an integer vocabulary from all unique token strings in column ``col``.

    Special tokens ``<PAD>`` (0) and ``<UNK>`` (1) are always added.

    Parameters
    ----------
    df : pd.DataFrame
        The curated dataset; ``col`` may hold JSON / repr lists or plain strings.
    col : str
        Column name containing token lists.

    Returns
    -------
    dict mapping token string → integer index.
    """
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for val in df[col].dropna():
        tokens = _parse_token_list(val)
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def _parse_token_list(value) -> List[str]:
    """Parse a column value that is either a list or a string repr of a list."""
    if isinstance(value, list):
        return [str(t) for t in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(t) for t in parsed]
    except (ValueError, SyntaxError):
        pass
    return [str(value)]


class HarmonicDataset:
    """
    PyTorch-compatible dataset for harmonic features derived from the
    Lakh-MSD curated DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``LakhMSDLinker.build_dataset()``.
    label_col : str
        Column to use as the target label (e.g., ``'primary_genre'``).
    mode : ``'classification'`` | ``'sequence'``
        * ``'classification'`` — yields (feature_vector, label_int)
        * ``'sequence'``       — yields (chord_token_ids, label_int)
    seq_col : str
        Column containing chord sequences (used in sequence mode).
        The value is expected to be a list or repr-list of Roman numeral strings.
    vocab : dict, optional
        Pre-built vocabulary. Built automatically if not supplied (sequence mode).
    max_seq_len : int
        Maximum sequence length (padded / truncated).
    feature_cols : list, optional
        Override the default ``NUMERIC_FEATURES`` list.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "primary_genre",
        mode: str = "classification",
        seq_col: str = "top_chords",
        vocab: Optional[Dict[str, int]] = None,
        max_seq_len: int = 128,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        self.mode        = mode
        self.max_seq_len = max_seq_len

        # ── Build label encoder ───────────────────────────────────────
        labels = df[label_col].fillna("unknown").astype(str).tolist()
        unique_labels = sorted(set(labels))
        self.label2idx: Dict[str, int] = {l: i for i, l in enumerate(unique_labels)}
        self.idx2label: Dict[int, str] = {i: l for l, i in self.label2idx.items()}
        self.labels     = [self.label2idx[l] for l in labels]
        self.num_classes = len(unique_labels)

        if mode == "classification":
            cols = feature_cols or NUMERIC_FEATURES
            avail = [c for c in cols if c in df.columns]
            self.feature_cols  = avail
            self.feature_matrix = df[avail].fillna(0.0).astype(float).values
            self.input_dim      = len(avail)

        elif mode == "sequence":
            self.vocab = vocab if vocab is not None else build_vocab(df, seq_col)
            self.vocab_size = len(self.vocab)
            self.sequences  = []
            for val in df[seq_col].fillna("[]"):
                tokens = _parse_token_list(val)
                ids    = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
                # Pad or truncate
                if len(ids) < max_seq_len:
                    ids = ids + [self.vocab["<PAD>"]] * (max_seq_len - len(ids))
                else:
                    ids = ids[:max_seq_len]
                self.sequences.append(ids)

        else:
            raise ValueError(f"Unknown mode: {mode!r}. Choose 'classification' or 'sequence'.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns a (input_tensor, label_tensor) pair.

        Requires PyTorch.  Import is deferred so the class can be imported
        without PyTorch installed (e.g., for inspection purposes).
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required: pip install torch") from exc

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.mode == "classification":
            x = torch.tensor(self.feature_matrix[idx], dtype=torch.float32)
        else:
            x = torch.tensor(self.sequences[idx], dtype=torch.long)

        return x, label

    def to_dataloader(self, batch_size: int = 32, shuffle: bool = True):
        """Convenience wrapper — returns a torch DataLoader."""
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise ImportError("torch is required: pip install torch") from exc
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def split(self, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
        """
        Split into train / val / test subsets using scikit-learn's StratifiedShuffleSplit.

        Returns
        -------
        train_idx, val_idx, test_idx — numpy arrays of integer indices.
        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

        n = len(self)
        indices = np.arange(n)
        labels  = np.array(self.labels)

        idx_train, idx_tmp = train_test_split(
            indices, test_size=val_frac + test_frac, stratify=labels, random_state=seed
        )
        val_ratio_of_tmp = val_frac / (val_frac + test_frac)
        idx_val, idx_test = train_test_split(
            idx_tmp, test_size=1.0 - val_ratio_of_tmp,
            stratify=labels[idx_tmp], random_state=seed
        )
        return idx_train, idx_val, idx_test
