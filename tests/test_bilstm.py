import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from src.models.bilstm import (  # noqa: E402
    PAD_ID,
    UNK_ID,
    BiLSTMClassifier,
    BiLSTMConfig,
    BiLSTMPredictor,
    TextClassificationDataset,
    Vocabulary,
    collate_batch,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    tokenize,
)
from src.training.train_bilstm import build_arg_parser, build_demo_dataframe, prepare_splits, train  # noqa: E402


def test_tokenizer_keeps_technical_terms():
    assert tokenize("Used C++, C# and Node.js") == ["used", "c++", "c#", "and", "node.js"]


def test_vocabulary_unknown_and_padding():
    vocab = Vocabulary.build(["python python sql", "python docker docker"], min_freq=2)
    assert vocab.stoi["<pad>"] == PAD_ID and vocab.stoi["<unk>"] == UNK_ID
    assert "python" in vocab.stoi and "docker" in vocab.stoi
    assert "sql" not in vocab.stoi  # below min_freq
    assert vocab.encode("sql python", 10)[0] == UNK_ID
    assert vocab.encode("", 10) == [UNK_ID]  # never empty
    assert len(vocab.encode("python " * 50, 5)) == 5  # truncation


def test_collate_pads_batch():
    vocab = Vocabulary.build(["a b c d e"], min_freq=1)
    dataset = TextClassificationDataset(["a b c d e", "a b"], [0, 1], vocab, max_len=10)
    padded, lengths, labels = collate_batch([dataset[0], dataset[1]])
    assert padded.shape == (2, 5)
    assert lengths.tolist() == [5, 2]
    assert padded[1, 2:].tolist() == [PAD_ID] * 3
    assert labels.tolist() == [0, 1]


def test_model_forward_shape():
    set_seed(0)
    model = BiLSTMClassifier(BiLSTMConfig(vocab_size=50, num_classes=3, embed_dim=8, hidden_dim=6))
    x = torch.randint(2, 50, (4, 7))
    lengths = torch.tensor([7, 5, 3, 1])
    assert model(x, lengths).shape == (4, 3)


def test_seed_makes_initialisation_reproducible():
    cfg = BiLSTMConfig(vocab_size=30, num_classes=2, embed_dim=4, hidden_dim=4)
    set_seed(7)
    a = BiLSTMClassifier(cfg).classifier.weight.clone()
    set_seed(7)
    b = BiLSTMClassifier(cfg).classifier.weight.clone()
    assert torch.equal(a, b)


def test_demo_training_checkpoint_roundtrip(tmp_path):
    out = tmp_path / "demo.pt"
    args = build_arg_parser().parse_args(
        ["--demo", "--epochs", "3", "--batch-size", "32", "--max-len", "60", "--output", str(out), "--cpu"]
    )
    metrics = train(args)
    assert out.is_file()
    assert 0.0 <= metrics["test_accuracy"] <= 1.0

    predictor = BiLSTMPredictor(out)
    assert predictor.is_demo  # synthetic checkpoints are flagged and hidden in the app
    assert predictor.label_names == ["data_science", "devops", "web_development"]
    probabilities = predictor.predict_proba(["docker kubernetes terraform aws", "react html css javascript"])
    assert probabilities.shape == (2, 3)
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
    top = predictor.predict(["docker kubernetes terraform aws"], top_k=1)[0][0]
    assert top[0] in predictor.label_names


def test_stratified_splits_have_no_overlap():
    df = build_demo_dataframe(samples_per_class=40)
    train_df, val_df, test_df, labels = prepare_splits(df, seed=1)
    assert len(labels) == 3
    assert not set(train_df["text"]) & set(test_df["text"])
    assert not set(train_df["text"]) & set(val_df["text"])


def test_load_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("does_not_exist.pt")


def test_dataloader_batches():
    vocab = Vocabulary.build(["a b c"], min_freq=1)
    dataset = TextClassificationDataset(["a b", "b c", "a"], [0, 1, 0], vocab, 5)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_batch)))
    assert batch[0].shape[0] == 3
