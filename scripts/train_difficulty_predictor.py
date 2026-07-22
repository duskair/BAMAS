import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eapae_agent_sys.planning.difficulty_predictor import DifficultyPredictor2


def resolve_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate

    fallback_candidates = [
        REPO_ROOT / "data/processed/offline_rl_dataset.jsonl",
        REPO_ROOT / "data/processed/math/offline_rl_dataset.jsonl",
        REPO_ROOT / "data/processed/mbpp/offline_rl_dataset.jsonl",
    ]
    for fallback in fallback_candidates:
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not find dataset: {dataset_path}")


def load_training_pairs(dataset_path: Path, max_samples: int = -1) -> Tuple[List[str], List[float]]:
    prompts_by_text: Dict[str, List[dict]] = defaultdict(list)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}: {exc}") from exc

            prompt = entry.get("task_description") or entry.get("prompt") or entry.get("question") or entry.get("text")
            if not prompt:
                continue

            prompts_by_text[str(prompt)].append(entry)
            if max_samples > 0 and sum(len(items) for items in prompts_by_text.values()) > max_samples:
                break

    prompts: List[str] = []
    labels: List[float] = []
    for prompt, entries in prompts_by_text.items():
        if max_samples > 0 and len(prompts) >= max_samples:
            break
        success_rate = sum(1.0 for entry in entries if entry.get("is_correct")) / float(len(entries))
        difficulty_label = max(0.0, min(1.0, 1.0 - success_rate))
        prompts.append(prompt)
        labels.append(difficulty_label)

    if not prompts:
        raise ValueError(f"No usable prompts were found in {dataset_path}")

    return prompts, labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BAMAS difficulty predictor regression head")
    parser.add_argument("--dataset", default="data/processed/math/offline_rl_dataset.jsonl", help="Path to the offline RL cache JSONL file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of prompts to train on (-1 = all)")
    parser.add_argument("--device", default="cpu", help="Training device (cpu or cuda)")
    parser.add_argument("--model-src", default="sentence-transformers/all-mpnet-base-v2", help="Sentence-transformers model to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="outputs/checkpoints/difficulty_predictor2_full.pt", help="Path for the saved checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    dataset_path = resolve_dataset_path(args.dataset)
    prompts, labels = load_training_pairs(dataset_path, max_samples=args.max_samples)
    print(f"Loaded {len(prompts)} prompts from {dataset_path}")

    predictor = DifficultyPredictor2(
        device=device,
        freeze_encoder=True,
        prepend_template=True,
        model_src=args.model_src,
    )
    predictor.to(device)

    optimizer = optim.Adam(predictor.reg_head.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    texts = prompts
    targets = torch.tensor(labels, dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        predictor.train()
        epoch_loss = 0.0
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]
            batch_targets = targets[start:start + args.batch_size]
            optimizer.zero_grad()
            preds = predictor(batch_texts, apply_sigmoid=True)
            loss = loss_fn(preds, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_texts)

        avg_loss = epoch_loss / len(texts)
        print(f"Epoch {epoch + 1}/{args.epochs} | mse={avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_full(output_path)
    print(f"Training complete. Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
