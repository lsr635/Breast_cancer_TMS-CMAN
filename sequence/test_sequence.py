#!/usr/bin/env python3
"""Batch test script for TMS-CMAN sequence classifier.

This script loads a trained model checkpoint, runs inference on the test split,
collects per-case outputs (probabilities, predicted labels), and packages each
case's raw frames together with a prediction report under a dedicated folder.
A consolidated CSV report is also produced with an "is_correct" flag so that
misclassified cases can be inspected quickly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TMSSequenceFolder, default_build_transforms
from train_s import InputAdapter
from models import TMS_CMAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TMS-CMAN on the test split and archive the outputs.")
    parser.add_argument("--path", type=str, required=True, help="Dataset root containing train/val/test splits.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained model state_dict (e.g. model/tms_cman/tms_cman_best_auc.pth).")
    parser.add_argument("--output", type=str, default="test_results", help="Destination folder for packed case results.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--img-size", type=int, default=224, help="Resize dimension used during training.")
    parser.add_argument("--temporal", type=int, default=1, help="1 to use temporal sequences (default), 0 for single-frame mode.")
    parser.add_argument("--max-T", type=int, default=9, help="Number of frames per sequence expected by the checkpoint.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device identifier (e.g. 'cuda:0' or 'cpu').")
    parser.add_argument("--threshold", type=float, default=None, help="Optional probability threshold for positive class (binary only). If omitted, tries to read JSON saved alongside the weights; falls back to argmax.")
    parser.add_argument("--threshold-json", type=str, default=None, help="Optional path to a JSON file storing {'best_thr': value}. Overrides --threshold if present.")
    parser.add_argument("--seq-source", type=str, default="exp1_vibrant", choices=["exp1_vibrant", "exp2_sub", "exp2_pre_sub", "exp2_17"], help="Sequence assembly mode used during training.")
    parser.add_argument("--tta", action="store_true", help="Enable simple horizontal flip test-time augmentation.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output directory if it already exists.")
    parser.add_argument("--copy-mode", choices=["copy", "link", "none"], default="copy", help="How to archive raw frames per case: copy files, create hard links (same filesystem), or skip saving frames.")
    return parser.parse_args()


def ensure_output_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Output directory '{path}' already exists. Use --overwrite to replace it.")
    os.makedirs(path, exist_ok=True)


def build_dataset(args: argparse.Namespace) -> TMSSequenceFolder:
    transform = default_build_transforms(train=False, img_size=args.img_size)
    dataset = TMSSequenceFolder(
        root=args.path,
        phase="test",
        img_size=args.img_size,
        temporal=bool(args.temporal),
        max_T=args.max_T,
        transform=transform,
        seq_source=args.seq_source,
        enable_aug=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("The test split is empty. Please confirm the dataset structure under path/test.")
    return dataset


def build_collate_fn():
    def _collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict]]):
        data, labels, metas = zip(*batch)
        data = torch.stack(data, dim=0)
        labels = torch.stack(labels, dim=0)
        return data, labels, list(metas)

    return _collate


def detect_in_channels(sample_tensor: torch.Tensor) -> int:
    if sample_tensor.dim() == 4:
        return int(sample_tensor.shape[1])
    if sample_tensor.dim() == 5:
        return int(sample_tensor.shape[2])
    raise RuntimeError(f"Unexpected tensor shape from dataset: {tuple(sample_tensor.shape)}")


def build_model(args: argparse.Namespace, in_channels: int, device: torch.device) -> torch.nn.Module:
    input_adapter = InputAdapter(in_channels=in_channels, out_channels=3)
    core = TMS_CMAN(num_classes=args.num_classes, img_size=args.img_size, pretrained=False, T_max=args.max_T)
    model = torch.nn.Sequential()
    model.add_module("input_adapter", input_adapter)
    model.add_module("net", core)

    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def resolve_threshold(args: argparse.Namespace) -> float | None:
    if args.threshold_json:
        with open(args.threshold_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("best_thr"))
    if args.threshold is not None:
        return args.threshold

    cand = os.path.join(os.path.dirname(args.weights), "tms_cman_best_auc_thr.json")
    if os.path.exists(cand):
        try:
            with open(cand, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("best_thr"))
        except Exception:
            pass
    return None


def format_case_id(meta_key: Tuple[str, str, int], fallback_path: str) -> str:
    if meta_key is None:
        base = os.path.splitext(os.path.basename(fallback_path))[0]
        return base
    bm, case, pidx = meta_key
    return f"{bm}_case{case}_p{int(pidx):02d}"


def archive_case(case_dir: str, copy_mode: str, frame_paths: Sequence[str]) -> None:
    if copy_mode == "none":
        return
    for src in frame_paths:
        if not os.path.isfile(src):
            continue
        dst = os.path.join(case_dir, os.path.basename(src))
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        elif copy_mode == "link":
            try:
                os.link(src, dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)


def write_case_report(case_dir: str, payload: Dict) -> None:
    out_path = os.path.join(case_dir, "result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_summary_csv(rows: List[Dict], csv_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def decide_prediction(probs: torch.Tensor, num_classes: int, threshold: float | None) -> int:
    if num_classes == 2 and threshold is not None:
        return int(probs[1] >= threshold)
    return int(torch.argmax(probs).item())


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith("cpu") else "cpu")
    ensure_output_dir(args.output, args.overwrite)

    dataset = build_dataset(args)
    collate_fn = build_collate_fn()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    sample_tensor, _, _ = dataset[0]
    in_channels = detect_in_channels(sample_tensor)
    model = build_model(args, in_channels=in_channels, device=device)

    threshold = resolve_threshold(args)
    if threshold is not None:
        print(f"[INFO] Using decision threshold {threshold:.4f} for the positive class.")
    else:
        print("[INFO] No threshold provided; using argmax over class probabilities.")

    class_names = dataset.classes
    os.makedirs(args.output, exist_ok=True)

    summary_rows: List[Dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", ncols=100):
            data, labels, metas = batch
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            if args.tta:
                logits = (logits + model(torch.flip(data, dims=[-1]))) / 2.0
            probs = torch.softmax(logits, dim=1)

            for i in range(data.size(0)):
                meta = metas[i]
                frame_paths: Sequence[str] = meta.get("paths", [])
                first_path = frame_paths[0] if frame_paths else ""
                case_id = format_case_id(meta.get("key"), first_path)
                case_dir = os.path.join(args.output, case_id)
                os.makedirs(case_dir, exist_ok=True)

                prob_vec = probs[i].detach().cpu()
                gt_idx = int(labels[i].item())
                pred_idx = decide_prediction(prob_vec, args.num_classes, threshold)
                gt_name = class_names[gt_idx] if gt_idx < len(class_names) else str(gt_idx)
                pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

                archive_case(case_dir, args.copy_mode, frame_paths)

                payload = {
                    "case_id": case_id,
                    "frame_paths": frame_paths,
                    "ground_truth": {
                        "index": gt_idx,
                        "label": gt_name,
                    },
                    "prediction": {
                        "index": pred_idx,
                        "label": pred_name,
                        "probabilities": [float(x) for x in prob_vec.tolist()],
                        "positive_score": float(prob_vec[1].item()) if prob_vec.numel() > 1 else float(prob_vec[0].item()),
                        "threshold": threshold,
                    },
                    "is_correct": bool(pred_idx == gt_idx),
                }
                write_case_report(case_dir, payload)

                summary_rows.append({
                    "case_id": case_id,
                    "ground_truth_idx": gt_idx,
                    "ground_truth_label": gt_name,
                    "pred_idx": pred_idx,
                    "pred_label": pred_name,
                    "probabilities": ";".join(f"{p:.6f}" for p in prob_vec.tolist()),
                    "positive_score": f"{prob_vec[1].item():.6f}" if prob_vec.numel() > 1 else f"{prob_vec[0].item():.6f}",
                    "is_correct": int(pred_idx == gt_idx),
                    "frame_count": len(frame_paths),
                    "frame_files": ";".join(os.path.basename(p) for p in frame_paths),
                    "case_dir": case_dir,
                })

    summary_csv = os.path.join(args.output, "summary.csv")
    save_summary_csv(summary_rows, summary_csv)
    print(f"[INFO] Summary written to: {summary_csv}")

    mistakes = [row for row in summary_rows if row["is_correct"] == 0]
    if mistakes:
        mistakes_csv = os.path.join(args.output, "misclassified.csv")
        save_summary_csv(mistakes, mistakes_csv)
        print(f"[INFO] Misclassifications saved to: {mistakes_csv}")
    else:
        print("[INFO] All test cases classified correctly under current decision rule.")


if __name__ == "__main__":
    main()
