"""Train the BiLSTM text classifier.

Run from the project root:

    # 1) Real data: any CSV with a text column and a label column
    python -m src.training.train_bilstm --csv data/raw/resumes.csv --text-col Resume --label-col Category

    # 2) Pipeline smoke test on SYNTHETIC data (checkpoint is flagged as demo and is
    #    never used for predictions in the app)
    python -m src.training.train_bilstm --demo --epochs 5

WHAT THE LABELS MEAN (read before trusting the model)
-----------------------------------------------------
The script prints the label distribution before training. A column such as
"Category" (e.g. 'Data Science', 'HR') is a *resume category*. A model trained on
it predicts the category - it does NOT measure how well a candidate fits a job
description. Only pass ``--labels-are-relevance`` if your CSV really contains
resume-job relevance judgements; the flag just records that in the checkpoint
metadata, it does not create or alter labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.models.bilstm import (
    BiLSTMClassifier,
    BiLSTMConfig,
    TextClassificationDataset,
    Vocabulary,
    collate_batch,
    save_checkpoint,
    set_seed,
)
from src.preprocessing import clean_text

DEFAULT_CHECKPOINT = Path("models_saved/bilstm_resume_category.pt")
DEMO_CHECKPOINT = Path("models_saved/bilstm_demo_synthetic.pt")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_demo_dataframe(samples_per_class: int = 120, seed: int = 42) -> pd.DataFrame:
    """SYNTHETIC text for testing the training pipeline end to end.

    The classes ('data_science', 'web_development', 'devops') are defined by
    keyword templates written by the project author. The data is artificial: results on it
    say nothing about real resumes.
    """
    rng = random.Random(seed)
    vocab_by_class = {
        "data_science": [
            "python", "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "nlp", "statistics",
            "regression", "classification", "feature", "engineering", "model", "training", "dataset",
            "jupyter", "visualization", "xgboost", "deep", "learning",
        ],
        "web_development": [
            "javascript", "react", "html", "css", "node.js", "typescript", "frontend", "backend", "rest",
            "api", "django", "flask", "responsive", "ui", "components", "express", "webpack", "sql", "web", "pages",
        ],
        "devops": [
            "docker", "kubernetes", "terraform", "aws", "jenkins", "ci/cd", "pipeline", "linux", "monitoring",
            "ansible", "deployment", "infrastructure", "bash", "cloud", "prometheus", "automation", "helm",
            "containers", "scaling", "reliability",
        ],
    }
    shared = ["experience", "project", "team", "developed", "responsible", "built", "improved", "worked", "using", "with"]
    rows = []
    for label, words in vocab_by_class.items():
        for _ in range(samples_per_class):
            length = rng.randint(25, 60)
            tokens = [rng.choice(words) if rng.random() < 0.55 else rng.choice(shared + sum(vocab_by_class.values(), [])) for _ in range(length)]
            rows.append({"text": " ".join(tokens), "label": label})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def load_dataframe(csv_path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in CSV. Available columns: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def inspect_labels(df: pd.DataFrame) -> None:
    """Print what the labels are, so the user can judge what the model will learn."""
    counts = df["label"].value_counts()
    print("\n=== Label inspection ===")
    print(f"Rows: {len(df)} | Distinct labels: {counts.size}")
    print(counts.to_string())
    print("Sample texts per label (first 120 characters):")
    for label in counts.index[:10]:
        example = df.loc[df["label"] == label, "text"].iloc[0][:120].replace("\n", " ")
        print(f"  [{label}] {example}...")
    print("========================\n")


def prepare_splits(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Deduplicate, drop tiny classes, then make a stratified 70/15/15 split."""
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)  # duplicates would leak across splits
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate texts to avoid train/test leakage.")
    counts = df["label"].value_counts()
    small = counts[counts < 5].index.tolist()
    if small:
        print(f"Dropping classes with fewer than 5 samples: {small}")
        df = df[~df["label"].isin(small)].reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("Need at least two classes with 5+ samples each to train a classifier.")

    label_names = sorted(df["label"].unique().tolist())
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), label_names


# ---------------------------------------------------------------------------
# Train / evaluate loops
# ---------------------------------------------------------------------------
def run_epoch(
    model: BiLSTMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates.

    Returns (average loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, lengths, labels in loader:
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict_labels(model: BiLSTMClassifier, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for inputs, lengths, labels in loader:
        logits = model(inputs.to(device), lengths.to(device))
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    if args.demo:
        df = build_demo_dataframe(seed=args.seed)
        source = "synthetic demo data generated by build_demo_dataframe()"
    else:
        if not args.csv:
            raise SystemExit("Provide --csv PATH (with --text-col / --label-col) or use --demo.")
        df = load_dataframe(Path(args.csv), args.text_col, args.label_col)
        source = str(args.csv)

    inspect_labels(df)
    train_df, val_df, test_df, label_names = prepare_splits(df, args.seed)
    label_to_id = {name: i for i, name in enumerate(label_names)}
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    vocab = Vocabulary.build(train_df["text"].tolist(), min_freq=args.min_freq, max_size=args.max_vocab)
    print(f"Vocabulary size (train only): {len(vocab)}")

    def make_loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = TextClassificationDataset(
            frame["text"].tolist(), [label_to_id[x] for x in frame["label"]], vocab, args.max_len
        )
        generator = torch.Generator().manual_seed(args.seed)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_batch, generator=generator)

    train_loader, val_loader, test_loader = make_loader(train_df, True), make_loader(val_df, False), make_loader(test_df, False)

    config = BiLSTMConfig(
        vocab_size=len(vocab),
        num_classes=len(label_names),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len,
    )
    model = BiLSTMClassifier(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(float(value))
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no validation improvement for {args.patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # evaluate the best-validation model, not the last one

    y_true, y_pred = predict_labels(model, test_loader, device)
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "best_val_loss": float(best_val_loss),
        "epochs_run": len(history["train_loss"]),
        "num_train": len(train_df),
        "num_val": len(val_df),
        "num_test": len(test_df),
    }
    print("\n=== Held-out TEST metrics ===")
    print(f"Accuracy: {metrics['test_accuracy']:.4f} | Macro-F1: {metrics['test_macro_f1']:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    dataset_info = {
        "source": source,
        "is_demo": bool(args.demo),
        "task": "relevance" if args.labels_are_relevance else "resume_category_classification",
        "label_meaning": (
            "resume-job relevance (as stated by the user)"
            if args.labels_are_relevance
            else "category label of the training CSV; NOT a measure of candidate suitability"
        ),
        "num_samples_after_cleaning": int(len(df)),
        "text_col": args.text_col,
        "label_col": args.label_col,
    }

    out_path = Path(args.output) if args.output else (DEMO_CHECKPOINT if args.demo else DEFAULT_CHECKPOINT)
    save_checkpoint(out_path, model, vocab, label_names, metrics, history, dataset_info, args.seed)
    history_path = out_path.with_suffix(".history.json")
    history_path.write_text(json.dumps({"history": history, "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"Checkpoint saved to: {out_path}")
    print(f"Loss/accuracy history saved to: {history_path}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the HireSense BiLSTM text classifier.")
    p.add_argument("--csv", help="Path to a CSV file with a text and a label column.")
    p.add_argument("--text-col", default="Resume")
    p.add_argument("--label-col", default="Category")
    p.add_argument("--demo", action="store_true", help="Train on SYNTHETIC data to test the pipeline only.")
    p.add_argument("--labels-are-relevance", action="store_true", help="Only if the CSV truly holds resume-job relevance labels.")
    p.add_argument("--output", help="Checkpoint path (.pt).")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=400)
    p.add_argument("--max-vocab", type=int, default=20000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        train(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
