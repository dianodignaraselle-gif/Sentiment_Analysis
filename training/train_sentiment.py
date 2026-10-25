"""Fine-tune XLM-RoBERTa on the upgraded dataset and save it for the app.

    python training/build_dataset.py      # once, creates training/data/
    python training/train_sentiment.py    # trains, evaluates, saves

Runs on CPU (≈15-25 min on a dual-core laptop) or a Colab GPU (a few minutes):
upload the project's training/ and backend/preprocessing.py, keeping the
same folder layout, then `!pip install -q transformers==4.46.0 sentencepiece
scikit-learn pandas openpyxl` and run the two commands above.

Outputs (default --out backend/models/foundation-sentiment-xlmr):
  model.safetensors, config.json, tokenizer files   <- loaded by the app
  training_info.json                                <- settings + metrics
  evaluation.md                                     <- report on the real-comment test set

Text goes through backend/preprocessing.py (clean_text + MAX_LENGTH), the
same module the app imports, so training and inference see identical input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "backend"))
from preprocessing import MAX_LENGTH, clean_text  # noqa: E402

BASE_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# The Hub checkpoint is pytorch_model.bin only, which transformers>=4.50
# refuses to load on torch<2.6. A safetensors copy made once with
#   AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).save_pretrained(dir)
# (plus the tokenizer) in training/cache/base-model is used instead when present.
_LOCAL_BASE = ROOT / "cache" / "base-model"
DEFAULT_BASE = str(_LOCAL_BASE) if (_LOCAL_BASE / "config.json").exists() else BASE_MODEL
LABELS = ["negative", "neutral", "positive"]  # same ids as the base checkpoint's head
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--out", default=str(PROJECT / "backend" / "models" / "foundation-sentiment-xlmr"))
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2, help="stop after this many epochs without validation gain")
    p.add_argument("--train-embeddings", action="store_true",
                   help="also fine-tune the 192M-parameter embedding matrix (needs ~3GB more RAM; "
                        "frozen by default - with ~1.5k examples it mostly adds overfitting)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(data_dir: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / f"{name}.csv")
    df["text"] = df["text"].astype(str).map(clean_text)
    df["y"] = df["label"].map(LABEL2ID)
    assert df["y"].notna().all(), f"unknown label in {name}.csv"
    return df


def make_batches(encodings: list, labels: list, batch_size: int, shuffle: bool, rng: random.Random):
    """Group similar-length examples to minimise padding (big CPU speed-up)."""
    order = list(range(len(encodings)))
    if shuffle:
        rng.shuffle(order)
    chunk = batch_size * 50
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: len(encodings[i]))
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(idx, encodings, labels, pad_id, device):
    width = max(len(encodings[i]) for i in idx)
    ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for row, i in enumerate(idx):
        ids[row, : len(encodings[i])] = torch.tensor(encodings[i])
        mask[row, : len(encodings[i])] = 1
    y = torch.tensor([labels[i] for i in idx], dtype=torch.long)
    return ids.to(device), mask.to(device), y.to(device)


@torch.no_grad()
def predict(model, encodings, pad_id, device, batch_size=32) -> np.ndarray:
    model.eval()
    probs = np.zeros((len(encodings), len(LABELS)), dtype=np.float32)
    dummy = [0] * len(encodings)
    for idx in make_batches(encodings, dummy, batch_size, shuffle=False, rng=random.Random(0)):
        ids, mask, _ = collate(idx, encodings, dummy, pad_id, device)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p, "recall_macro": r, "f1_macro": f}


def report_md(title: str, y_true, y_pred) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    lines = [f"### {title}", "", "```", classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=LABELS, digits=3, zero_division=0), "```", "",
        "Confusion matrix (rows = true, columns = predicted):", "",
        "| | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    lines += [f"| **{LABELS[i]}** | " + " | ".join(str(v) for v in cm[i]) + " |" for i in range(3)]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    print(f"device={device}  max_length={MAX_LENGTH}")

    train, val, test = (load_split(args.data, s) for s in ("train", "validation", "test"))
    print(f"train={len(train)} validation={len(val)} test={len(test)}")
    print("train labels:", train["label"].value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
    ).to(device)
    if not args.train_embeddings:
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False

    def encode(texts):
        return tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)["input_ids"]

    enc = {name: encode(df["text"]) for name, df in (("train", train), ("val", val), ("test", test))}
    pad_id = tokenizer.pad_token_id

    # Baseline: the stock checkpoint on the same test set, before any training.
    base_test_probs = predict(model, enc["test"], pad_id, device)
    base_test = metrics(test["y"], base_test_probs.argmax(1))
    print("base model on test:", {k: round(v, 3) for k, v in base_test.items()})

    counts = np.bincount(train["y"], minlength=3)
    class_weights = torch.tensor(len(train) / (3 * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in trainable if not any(k in n for k in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in trainable if any(k in n for k in no_decay)], "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    rng = random.Random(args.seed)
    best_f1, best_state, best_epoch, bad_epochs, history = -1.0, None, 0, 0, []
    y_train = train["y"].tolist()
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        batches = make_batches(enc["train"], y_train, args.batch_size, shuffle=True, rng=rng)
        for step, idx in enumerate(batches, 1):
            ids, mask, y = collate(idx, enc["train"], y_train, pad_id, device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 20 == 0 or step == len(batches):
                rate = (time.time() - started) / step
                print(f"  epoch {epoch} step {step}/{len(batches)} loss={running / step:.4f} "
                      f"({rate:.1f}s/step, epoch ETA {rate * (len(batches) - step) / 60:.1f} min)", flush=True)

        val_m = metrics(val["y"], predict(model, enc["val"], pad_id, device).argmax(1))
        history.append({"epoch": epoch, "train_loss": running / len(batches), **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"epoch {epoch}: train_loss={running / len(batches):.4f} "
              f"val_acc={val_m['accuracy']:.3f} val_f1_macro={val_m['f1_macro']:.3f}", flush=True)
        if val_m["f1_macro"] > best_f1:
            best_f1, best_epoch, bad_epochs = val_m["f1_macro"], epoch, 0
            best_state = {n: p.detach().cpu().clone() for n, p in trainable}  # frozen weights never change
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop: no validation gain for {args.patience} epochs")
                break

    model.load_state_dict(best_state, strict=False)
    print(f"restored best epoch {best_epoch} (val f1_macro={best_f1:.3f})")

    test_probs = predict(model, enc["test"], pad_id, device)
    test_pred = test_probs.argmax(1)
    test_m = metrics(test["y"], test_pred)
    print(classification_report(test["y"], test_pred, labels=[0, 1, 2], target_names=LABELS, digits=3, zero_division=0))
    print(pd.DataFrame(confusion_matrix(test["y"], test_pred, labels=[0, 1, 2]), index=LABELS, columns=LABELS))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    wrong = test.assign(pred=[LABELS[i] for i in test_pred], confidence=[round(float(c), 3) for c in test_probs.max(1)])
    wrong = wrong[wrong["pred"] != wrong["label"]]
    report = [
        "# Evaluation on held-out real comments", "",
        f"Test set: {len(test)} real, human-written comments (never used for training): "
        f"{test['label'].value_counts().reindex(LABELS).to_dict()}.", "",
        "| model | accuracy | precision (macro) | recall (macro) | F1 (macro) |", "|---|---|---|---|---|",
        "| stock `{}` | {accuracy:.3f} | {precision_macro:.3f} | {recall_macro:.3f} | {f1_macro:.3f} |".format(
            BASE_MODEL, **base_test),
        "| fine-tuned (this model) | {accuracy:.3f} | {precision_macro:.3f} | {recall_macro:.3f} | {f1_macro:.3f} |".format(
            **test_m), "",
        report_md("Fine-tuned model", test["y"], test_pred),
        report_md("Stock model (before)", test["y"], base_test_probs.argmax(1)),
        "### Misclassified test comments", "",
        "| true | predicted | conf. | comment |", "|---|---|---|---|",
    ] + [f"| {r.label} | {r.pred} | {r.confidence} | {r.text[:160].replace('|', '/')} |" for r in wrong.itertuples()]
    (out / "evaluation.md").write_text("\n".join(report) + "\n")
    info = {
        "base_model": BASE_MODEL,
        "labels": LABELS,
        "max_length": MAX_LENGTH,
        "preprocessing": "backend/preprocessing.py:clean_text",
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("data", "out")},
        "train_size": len(train), "validation_size": len(val), "test_size": len(test),
        "best_epoch": best_epoch, "history": history,
        "test_metrics": test_m, "base_model_test_metrics": base_test,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"), "device": device,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
    }
    (out / "training_info.json").write_text(json.dumps(info, indent=2))
    print(f"saved model + tokenizer + evaluation.md to {out}")


if __name__ == "__main__":
    main()
