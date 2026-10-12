"""BiLSTM text classifier in PyTorch (vocabulary, dataset, model, checkpoints, inference).

SCOPE WARNING: this model learns whatever labels it is trained on (for example
"resume category"). Unless it is trained on genuine resume-job relevance labels,
its output is a *category prediction* and must never be presented as candidate
suitability or merged into the recruitment relevance score.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
# Same idea as the TF-IDF token pattern: keeps c++, c#, node.js as single tokens.
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#]*(?:\.[a-z0-9]+)*")


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible (Python, NumPy, PyTorch CPU+GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Token <-> id mapping with <pad>=0 and <unk>=1."""

    def __init__(self, stoi: dict[str, int] | None = None) -> None:
        self.stoi: dict[str, int] = stoi or {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 2, max_size: int = 20000) -> "Vocabulary":
        """Build from TRAINING texts only (never include validation/test text)."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        stoi = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token, freq in counter.most_common():
            if freq < min_freq or len(stoi) >= max_size:
                break
            stoi[token] = len(stoi)
        return cls(stoi)

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> list[int]:
        """Tokens -> ids; unknown tokens -> UNK; truncated to max_len; never empty."""
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokenize(text)[:max_len]]
        return ids or [UNK_ID]

    def to_dict(self) -> dict[str, int]:
        return dict(self.stoi)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Vocabulary":
        return cls({str(k): int(v) for k, v in data.items()})


class TextClassificationDataset(Dataset):
    """Holds encoded texts and integer labels."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocabulary, max_len: int) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.encoded = [vocab.encode(t, max_len) for t in texts]
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        ids = self.encoded[index]
        return torch.tensor(ids, dtype=torch.long), len(ids), self.labels[index]


def collate_batch(batch: list[tuple[torch.Tensor, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch dynamically to its longest sequence (pad id = 0)."""
    sequences, lengths, labels = zip(*batch)
    padded = pad_sequence(list(sequences), batch_first=True, padding_value=PAD_ID)
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


@dataclass
class BiLSTMConfig:
    vocab_size: int
    num_classes: int
    embed_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    max_len: int = 400

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "max_len": self.max_len,
        }


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> concat(final fwd, final bwd hidden) -> dropout -> linear."""

    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim * 2, config.num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        # Packing makes the LSTM ignore padding when computing the final states.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # last layer: forward, backward
        return self.classifier(self.dropout(features))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: BiLSTMClassifier,
    vocab: Vocabulary,
    label_names: list[str],
    metrics: dict[str, Any],
    history: dict[str, list[float]],
    dataset_info: dict[str, Any],
    seed: int,
) -> None:
    """Save weights + vocabulary + label mapping + metadata in ONE file.

    ``label_names[i]`` is the human-readable class for output index ``i``. Keeping the
    mapping inside the checkpoint prevents wrong-label bugs at inference time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "vocab": vocab.to_dict(),
        "label_names": list(label_names),
        "metrics": metrics,
        "history": history,
        "dataset_info": dataset_info,
        "seed": int(seed),
        "torch_version": str(torch.__version__),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict (weights_only=True: refuses arbitrary pickled code)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"model_state", "model_config", "vocab", "label_names", "dataset_info"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    return checkpoint


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class BiLSTMPredictor:
    """Loads a checkpoint and predicts class probabilities for new texts."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        self.config = BiLSTMConfig(**checkpoint["model_config"])
        if len(checkpoint["label_names"]) != self.config.num_classes:
            raise ValueError("Label mapping size does not match the model's number of classes.")
        self.label_names: list[str] = list(checkpoint["label_names"])
        self.vocab = Vocabulary.from_dict(checkpoint["vocab"])
        self.dataset_info: dict[str, Any] = checkpoint["dataset_info"]
        self.metrics: dict[str, Any] = checkpoint.get("metrics", {})
        self.model = BiLSTMClassifier(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @property
    def is_demo(self) -> bool:
        return bool(self.dataset_info.get("is_demo", True))

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return array (n_texts, n_classes) of softmax probabilities."""
        if not texts:
            return np.zeros((0, self.config.num_classes))
        encoded = [self.vocab.encode(t, self.config.max_len) for t in texts]
        batch = [(torch.tensor(ids), len(ids), 0) for ids in encoded]
        padded, lengths, _ = collate_batch(batch)
        logits = self.model(padded.to(self.device), lengths.to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, texts: Sequence[str], top_k: int = 3) -> list[list[tuple[str, float]]]:
        """Top-k (label, probability) pairs for each text."""
        probabilities = self.predict_proba(texts)
        output = []
        for row in probabilities:
            order = np.argsort(-row)[:top_k]
            output.append([(self.label_names[i], float(row[i])) for i in order])
        return output
