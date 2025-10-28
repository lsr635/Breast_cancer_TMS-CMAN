"""
Dataset for dynamic contrast-enhanced (DCE) breast MRI sequences with
time-ordered frames (e.g., 1 pre-contrast + multiple post-contrast phases).

Key features:
    * Recursive image discovery per case (now supports class folders like benign/m)
    * Automatic label inference from class subdirectories when no CSV/JSON provided
    * Phase parsing & chronological reordering
    * Missing-frame padding (with masks)
    * Time-synchronized data augmentation
    * Optional per-case (pre-frame) normalization
    * Optional global normalization
    * Flexible resize strategies for very small images (keep ratio + padding)
    * Custom collate function for DataLoader integration
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode, RandomAffine
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _normalize_phase_key(token: str) -> str:
    return (
        token.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("+", "")
    )


def load_label_mapping(
    path: str,
    case_field: str = "case_id",
    label_field: str = "label",
    strict: bool = False,
) -> Dict[str, Union[int, float, str]]:
    if path is None:
        return {}

    ext = os.path.splitext(path)[1].lower()
    mapping: Dict[str, Union[int, float, str]] = {}

    if ext in {".csv", ".tsv"}:
        delimiter = "," if ext == ".csv" else "\t"
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            use_first = case_field not in (reader.fieldnames or [])
            use_last = label_field not in (reader.fieldnames or [])
            for row in reader:
                case_id = (
                    row.get(case_field)
                    if not use_first
                    else row.get(reader.fieldnames[0])
                )
                label_val = (
                    row.get(label_field)
                    if not use_last
                    else row.get(reader.fieldnames[-1])
                )
                if case_id is None or label_val is None or label_val == "":
                    continue
                case_id = case_id.strip()
                label_val = label_val.strip()
                try:
                    label_val = int(label_val)
                except ValueError:
                    try:
                        label_val = float(label_val)
                    except ValueError:
                        pass
                mapping[case_id] = label_val

    elif ext in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        if isinstance(content, dict):
            mapping = content
        elif isinstance(content, list):
            for entry in content:
                case_id = entry.get(case_field)
                label_val = entry.get(label_field)
                if case_id is None or label_val is None:
                    continue
                mapping[str(case_id)] = label_val
    return mapping


def _default_phase_aliases(phases: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for phase in phases:
        aliases[_normalize_phase_key(phase)] = phase

    aliases[_normalize_phase_key("pre")] = "pre"
    aliases[_normalize_phase_key("vibrant")] = "pre"

    for i in range(1, 33):
        canonical = f"c{i}"
        aliases[_normalize_phase_key(canonical)] = canonical
        aliases[_normalize_phase_key(f"vibrant+c{i}")] = canonical
        aliases[_normalize_phase_key(f"vibrantc{i}")] = canonical
        aliases[_normalize_phase_key(f"phase{i}")] = canonical

    return aliases


@dataclass
class DatasetStats:
    total_cases: int = 0


@dataclass
class CaseEntry:
    case_id: str
    file_paths: List[str]
    label: Optional[Union[int, float, str]] = None
    source_dir: Optional[str] = None


class TimeSynchronizedAugmenter:
    def __init__(
        self,
        flip_prob: float = 0.5,
        affine_prob: float = 0.8,
        max_rotate: float = 10.0,
        max_translate: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        max_shear: float = 5.0,
        color_prob: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.05,
        hue: float = 0.02,
        noise_std: float = 0.0,
        fill: float = 0.0,
    ) -> None:
        self.flip_prob = max(0.0, min(1.0, flip_prob))
        self.affine_prob = max(0.0, min(1.0, affine_prob))
        self.color_prob = max(0.0, min(1.0, color_prob))

        self.max_rotate = float(max_rotate)
        self.max_translate = float(max_translate)
        self.scale_range = scale_range
        self.max_shear = float(max_shear)

        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.noise_std = float(noise_std)
        self.fill = float(fill)

        self._affine_enabled = any(
            v > 0
            for v in (
                abs(self.max_rotate),
                abs(self.max_translate),
                abs(self.scale_range[0] - 1.0),
                abs(self.scale_range[1] - 1.0),
                abs(self.max_shear),
            )
        )
        self._color_enabled = any(
            v > 0 for v in (self.brightness, self.contrast, self.saturation, self.hue)
        )

    def __call__(self, frames: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(frames) == 0:
            return []

        augmented: List[torch.Tensor] = []
        do_flip = random.random() < self.flip_prob

        apply_affine = self._affine_enabled and (random.random() < self.affine_prob)
        if apply_affine:
            affine_params = RandomAffine.get_params(
                degrees=(-self.max_rotate, self.max_rotate),
                translate=(self.max_translate, self.max_translate),
                scale_ranges=self.scale_range,
                shears=(-self.max_shear, self.max_shear),
                img_size=frames[0].shape[-2:],
            )
        else:
            affine_params = None

        apply_color = self._color_enabled and (random.random() < self.color_prob)
        if apply_color:
            brightness = (
                (max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
                if self.brightness > 0
                else None
            )
            contrast = (
                (max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
                if self.contrast > 0
                else None
            )
            saturation = (
                (max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
                if self.saturation > 0
                else None
            )
            hue = (-self.hue, self.hue) if self.hue > 0 else None
            if all(v is None for v in (brightness, contrast, saturation, hue)):
                apply_color = False
                color_params = None
            else:
                color_params = ColorJitter.get_params(
                    brightness, contrast, saturation, hue
                )
        else:
            color_params = None

        for img in frames:
            out = img.clone()

            if do_flip:
                out = TF.hflip(out)

            if apply_affine and affine_params is not None:
                angle, translations, scale, shear = affine_params
                out = TF.affine(
                    out,
                    angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=self.fill,
                )

            if apply_color and color_params is not None:
                brightness_factor, contrast_factor, saturation_factor, hue_factor = (
                    color_params
                )
                out = TF.adjust_brightness(out, brightness_factor)
                out = TF.adjust_contrast(out, contrast_factor)
                out = TF.adjust_saturation(out, saturation_factor)
                out = TF.adjust_hue(out, hue_factor)
                out = torch.clamp(out, 0.0, 1.0)

            if self.noise_std > 0:
                noise = torch.randn_like(out) * self.noise_std
                out = torch.clamp(out + noise, 0.0, 1.0)

            augmented.append(out)

        return augmented


class DCESequenceDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        phases: Sequence[str] = ("pre", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"),
        case_ids: Optional[Sequence[str]] = None,
        label_map: Optional[Dict[str, Union[int, float, str]]] = None,
        image_size: Optional[Tuple[int, int]] = (96, 96),
        num_channels: int = 1,
        augment: bool = False,
        augmenter: Optional[TimeSynchronizedAugmenter] = None,
        per_case_pre_norm: bool = False,
        global_normalize: bool = False,
        normalize_mean: Sequence[float] = (0.5, 0.5, 0.5),
        normalize_std: Sequence[float] = (0.25, 0.25, 0.25),
        duplicate_missing: str = "previous",
        return_dict: bool = True,
        return_label: Optional[bool] = None,
        label_dtype: torch.dtype = torch.long,
        dtype: torch.dtype = torch.float32,
        antialias: bool = True,
        resize_mode: str = "keep_ratio_pad",
        padding_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.image_size = (
            tuple(int(v) for v in image_size) if image_size is not None else None
        )
        self.num_channels = int(num_channels)
        self.augment = bool(augment)
        self.per_case_pre_norm = bool(per_case_pre_norm)
        self.global_normalize = bool(global_normalize)
        self.normalize_mean = tuple(float(v) for v in normalize_mean)
        self.normalize_std = tuple(float(v) for v in normalize_std)
        self.duplicate_missing = duplicate_missing.lower()
        self.return_dict = bool(return_dict)
        self.dtype = dtype
        self.label_dtype = label_dtype
        self.antialias = antialias
        self.resize_mode = resize_mode.lower()
        self.padding_value = float(padding_value)

        self.phases = tuple(phases)
        self.phase_to_index = {phase: idx for idx, phase in enumerate(self.phases)}
        self.pre_phase_index = self.phase_to_index.get("pre", 0)

        self.phase_regex = re.compile(r"(pre|vibrant(?:\+?c\d+)?|c\d+)", re.I)
        self.slice_regex = re.compile(r"p[-_](?P<idx>\d+)", re.I)
        self.phase_aliases = _default_phase_aliases(self.phases)

        base_label_map = dict(label_map or {})
        (
            self.case_entries,
            inferred_label_map,
            self.class_to_label,
        ) = self._build_case_entries(
            root_dir=self.root_dir,
            case_ids=case_ids,
            base_label_map=base_label_map,
        )

        self.case_entries.sort(key=lambda entry: entry.case_id)
        self.case_ids = [entry.case_id for entry in self.case_entries]
        self.label_map = inferred_label_map

        if return_label is None:
            self.return_label = bool(self.label_map)
        else:
            self.return_label = bool(return_label)

        self.default_label_value: Union[int, float, str, None] = (
            0 if self.return_label else None
        )

        if augmenter is None and self.augment:
            self.augmenter = TimeSynchronizedAugmenter()
        else:
            self.augmenter = augmenter

        self.stats = DatasetStats(total_cases=len(self.case_ids))

    def _collect_case_images(self, case_dir: str) -> List[str]:
        if os.path.isfile(case_dir) and case_dir.lower().endswith(IMAGE_EXTENSIONS):
            return [case_dir]
        pattern = os.path.join(case_dir, "**", "*")
        files = [
            path
            for path in glob.glob(pattern, recursive=True)
            if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        ]
        return sorted(files)

    def _extract_case_id(self, path: str) -> str:
        basename = os.path.splitext(os.path.basename(path))[0]
        match = self.phase_regex.search(basename)
        if match:
            candidate = basename[: match.start()]
        else:
            candidate = basename
        candidate = candidate.strip("_- ")
        if not candidate:
            candidate = "case"
        return candidate

    def _group_files_by_case(
        self,
        file_paths: Sequence[str],
        prefix: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for path in sorted(file_paths):
            case_id = self._extract_case_id(path)
            if prefix:
                case_id = f"{prefix}_{case_id}"
            grouped.setdefault(case_id, []).append(path)
        return grouped

    def _build_case_entries(
        self,
        root_dir: str,
        case_ids: Optional[Sequence[str]],
        base_label_map: Dict[str, Union[int, float, str]],
    ) -> Tuple[
        List[CaseEntry],
        Dict[str, Union[int, float, str]],
        Dict[str, int],
    ]:
        entries: List[CaseEntry] = []
        label_map_out: Dict[str, Union[int, float, str]] = dict(base_label_map)
        class_to_label: Dict[str, int] = {}

        if case_ids is not None:
            for cid in case_ids:
                case_dir = os.path.join(root_dir, cid)
                files = self._collect_case_images(case_dir)
                if not files:
                    continue
                label = label_map_out.get(cid)
                entries.append(
                    CaseEntry(
                        case_id=cid,
                        file_paths=files,
                        label=label,
                        source_dir=case_dir if os.path.isdir(case_dir) else None,
                    )
                )
            return entries, label_map_out, class_to_label

        root_images = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, fname))
            and os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
        ]
        if root_images:
            grouped = self._group_files_by_case(root_images, prefix=None)
            for case_id, files in grouped.items():
                label = label_map_out.get(case_id)
                entries.append(
                    CaseEntry(
                        case_id=case_id,
                        file_paths=files,
                        label=label,
                        source_dir=root_dir,
                    )
                )
            return entries, label_map_out, class_to_label

        candidate_dirs = [
            d
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ]
        candidate_dirs.sort()
        if not candidate_dirs:
            return entries, label_map_out, class_to_label

        next_label = 0
        for dir_name in candidate_dirs:
            class_path = os.path.join(root_dir, dir_name)
            sub_case_dirs = [
                sub
                for sub in os.listdir(class_path)
                if os.path.isdir(os.path.join(class_path, sub)) and not sub.startswith(".")
            ]
            image_files = [
                os.path.join(class_path, fname)
                for fname in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, fname))
                and os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
            ]

            class_label = base_label_map.get(dir_name)
            if class_label is None:
                class_label = next_label
                next_label += 1
            class_to_label[dir_name] = class_label

            if sub_case_dirs:
                sub_case_dirs.sort()
                for sub in sub_case_dirs:
                    case_dir = os.path.join(class_path, sub)
                    files = self._collect_case_images(case_dir)
                    if not files:
                        continue
                    case_id = f"{dir_name}_{sub}"
                    label = base_label_map.get(case_id, class_label)
                    label_map_out[case_id] = label
                    entries.append(
                        CaseEntry(
                            case_id=case_id,
                            file_paths=files,
                            label=label,
                            source_dir=case_dir,
                        )
                    )
            elif image_files:
                grouped = self._group_files_by_case(image_files, prefix=dir_name)
                for case_id, files in grouped.items():
                    label = base_label_map.get(case_id, class_label)
                    label_map_out[case_id] = label
                    entries.append(
                        CaseEntry(
                            case_id=case_id,
                            file_paths=files,
                            label=label,
                            source_dir=class_path,
                        )
                    )

        return entries, label_map_out, class_to_label

    def _parse_phase_and_slice(self, path: str) -> Tuple[Optional[str], Optional[int]]:
        name = path.lower()
        phase_matches = list(self.phase_regex.finditer(name))
        if not phase_matches:
            return None, None

        token = phase_matches[-1].group(0)
        normalized = _normalize_phase_key(token)
        phase = self.phase_aliases.get(normalized)
        if phase not in self.phase_to_index:
            return None, None

        slice_match = self.slice_regex.search(name)
        slice_idx = int(slice_match.group("idx")) if slice_match else 0
        return phase, slice_idx

    def _group_and_select(
        self, file_paths: Sequence[str]
    ) -> Tuple[List[Optional[str]], List[int]]:
        buckets: Dict[str, List[Tuple[int, str]]] = {phase: [] for phase in self.phases}
        for path in file_paths:
            phase, slice_idx = self._parse_phase_and_slice(path)
            if phase is None:
                continue
            buckets[phase].append((slice_idx, path))

        ordered_paths: List[Optional[str]] = []
        presence_flags: List[int] = []
        for phase in self.phases:
            entries = sorted(buckets[phase], key=lambda x: x[0])
            if not entries:
                ordered_paths.append(None)
                presence_flags.append(0)
            else:
                ordered_paths.append(entries[0][1])
                presence_flags.append(1)

        return ordered_paths, presence_flags

    def _read_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("L")
            tensor = TF.to_tensor(img)
        if self.num_channels == 1:
            return tensor.to(self.dtype)
        if tensor.shape[0] == self.num_channels:
            return tensor.to(self.dtype)
        if tensor.shape[0] == 1 and self.num_channels > 1:
            tensor = tensor.repeat(self.num_channels, 1, 1)
        elif tensor.shape[0] > self.num_channels:
            tensor = tensor[: self.num_channels]
        else:
            tensor = tensor.repeat(self.num_channels, 1, 1)
        return tensor.to(self.dtype)

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_size is None:
            return tensor

        target_h, target_w = self.image_size
        if self.resize_mode == "stretch":
            return TF.resize(
                tensor,
                size=self.image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=self.antialias,
            )

        h, w = tensor.shape[-2:]
        scale = min(target_h / h, target_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = TF.resize(
            tensor,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=self.antialias,
        )

        if self.resize_mode == "keep_ratio_crop":
            pad_h = max(0, target_h - new_h)
            pad_w = max(0, target_w - new_w)
            padded = TF.pad(
                resized,
                padding=(
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - pad_w // 2,
                    pad_h - pad_h // 2,
                ),
                fill=self.padding_value,
            )
            cropped = TF.center_crop(padded, output_size=[target_h, target_w])
            return cropped.to(self.dtype)

        pad_h = max(0, target_h - new_h)
        pad_w = max(0, target_w - new_w)
        padded = TF.pad(
            resized,
            padding=(
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            ),
            fill=self.padding_value,
        )
        return padded.to(self.dtype)

    def _fill_missing(
        self,
        frames: List[Optional[torch.Tensor]],
        presence_flags: List[int],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        filled: List[torch.Tensor] = []
        num_phases = len(frames)

        if self.image_size is not None:
            height, width = self.image_size
        else:
            first_available = next((f for f in frames if f is not None), None)
            if first_available is None:
                raise RuntimeError(
                    "All phases missing for this case and no target image_size specified."
                )
            height, width = first_available.shape[-2:]

        zero_template = torch.zeros(
            (self.num_channels, height, width),
            dtype=self.dtype,
        )

        for idx, frame in enumerate(frames):
            if frame is not None:
                filled.append(frame)
                continue

            if self.duplicate_missing == "previous":
                if filled:
                    filled.append(filled[-1].clone())
                else:
                    filled.append(zero_template.clone())
            elif self.duplicate_missing == "zero":
                filled.append(zero_template.clone())
            elif self.duplicate_missing == "nearest":
                offset = 1
                replacement = None
                while offset < num_phases:
                    left = idx - offset
                    right = idx + offset
                    if left >= 0 and frames[left] is not None:
                        replacement = frames[left]
                        break
                    if right < num_phases and frames[right] is not None:
                        replacement = frames[right]
                        break
                    offset += 1
                if replacement is not None:
                    filled.append(replacement.clone())
                elif filled:
                    filled.append(filled[-1].clone())
                else:
                    filled.append(zero_template.clone())
            else:
                filled.append(zero_template.clone())

        return filled, presence_flags

    def _per_case_normalize(
        self, frames: List[torch.Tensor], presence: List[int]
    ) -> List[torch.Tensor]:
        if len(frames) == 0:
            return frames

        reference_idx = self.pre_phase_index
        if presence[reference_idx] == 0:
            available_indices = [i for i, flag in enumerate(presence) if flag == 1]
            if not available_indices:
                return frames
            reference_idx = available_indices[0]

        ref = frames[reference_idx]
        mean = ref.mean(dim=(-2, -1), keepdim=True)
        std = ref.std(dim=(-2, -1), keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)

        normalized = [(img - mean) / std for img in frames]
        return normalized

    def _global_normalize(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            TF.normalize(img, mean=self.normalize_mean, std=self.normalize_std)
            for img in frames
        ]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        entry = self.case_entries[idx]
        case_id = entry.case_id
        file_paths = entry.file_paths

        if not file_paths:
            raise RuntimeError(f"No valid images found for case_id={case_id}")

        ordered_paths, presence_flags = self._group_and_select(file_paths)

        raw_frames: List[Optional[torch.Tensor]] = []
        for path in ordered_paths:
            if path is None:
                raw_frames.append(None)
            else:
                tensor = self._read_image(path)
                tensor = self._resize(tensor)
                raw_frames.append(tensor)

        filled_frames, presence_flags = self._fill_missing(raw_frames, presence_flags)

        if self.augment and self.augmenter is not None:
            filled_frames = self.augmenter(filled_frames)

        if self.per_case_pre_norm:
            filled_frames = self._per_case_normalize(filled_frames, presence_flags)

        if self.global_normalize:
            filled_frames = self._global_normalize(filled_frames)

        sequence = torch.stack(filled_frames, dim=0)
        mask = torch.tensor(presence_flags, dtype=torch.float32)

        label_value: Optional[Union[int, float, str]] = self.label_map.get(case_id)
        if label_value is None and entry.label is not None:
            label_value = entry.label

        if self.return_label:
            if label_value is None:
                label_tensor: Optional[torch.Tensor] = torch.zeros(
                    (), dtype=self.label_dtype
                )
            elif isinstance(label_value, torch.Tensor):
                label_tensor = label_value.to(self.label_dtype)
            else:
                label_tensor = torch.tensor(label_value, dtype=self.label_dtype)
        else:
            label_tensor = None

        sample_paths = ordered_paths

        if self.return_dict:
            sample = {
                "images": sequence,
                "mask": mask,
                "case_id": case_id,
                "paths": sample_paths,
            }
            if self.return_label:
                sample["label"] = label_tensor
            return sample

        return sequence, label_tensor, mask, case_id, sample_paths

    def describe(self) -> str:
        summary = [
            f"DCESequenceDataset summary:",
            f"  Root directory : {self.root_dir}",
            f"  Cases          : {len(self.case_ids)}",
            f"  Phases         : {self.phases}",
            f"  Image size     : {self.image_size}",
            f"  Num channels   : {self.num_channels}",
            f"  Resize mode    : {self.resize_mode}",
            f"  Augment        : {self.augment}",
            f"  Per-case norm  : {self.per_case_pre_norm}",
            f"  Global normalize: {self.global_normalize}",
            f"  Return dict    : {self.return_dict}",
        ]
        if self.class_to_label:
            summary.append(f"  Classes        : {self.class_to_label}")
        if self.return_label:
            summary.append(f"  Label dtype    : {self.label_dtype}")
            summary.append(f"  Labels loaded  : {len(self.label_map)}")
        return "\n".join(summary)


def dce_collate_fn(
    batch: Sequence[Union[Dict[str, torch.Tensor], Tuple]]
) -> Dict[str, Union[torch.Tensor, List[str], List[List[Optional[str]]]]]:
    if len(batch) == 0:
        return {}

    first = batch[0]
    if isinstance(first, dict):
        images = torch.stack([item["images"] for item in batch], dim=0)
        mask = torch.stack([item["mask"] for item in batch], dim=0)
        case_ids = [item["case_id"] for item in batch]
        paths = [item["paths"] for item in batch]

        batch_dict = {
            "images": images,
            "mask": mask,
            "case_id": case_ids,
            "paths": paths,
        }
        if "label" in first and first["label"] is not None:
            labels = torch.stack([item["label"] for item in batch], dim=0)
            batch_dict["label"] = labels
        else:
            batch_dict["label"] = None
        return batch_dict

    sequences, labels, masks, case_ids, paths = zip(*batch)
    images = torch.stack(sequences, dim=0)
    mask_tensor = torch.stack(masks, dim=0)
    case_id_list = list(case_ids)
    path_list = list(paths)

    if labels[0] is None:
        label_tensor = None
    else:
        label_tensor = torch.stack(labels, dim=0)

    return {
        "images": images,
        "label": label_tensor,
        "mask": mask_tensor,
        "case_id": case_id_list,
        "paths": path_list,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview DCESequenceDataset samples.")
    parser.add_argument("--root", "-r", required=True, help="Path to dataset root directory.")
    parser.add_argument("--labels", "-l", default=None, help="Optional CSV/TSV/JSON label file.")
    parser.add_argument(
        "--cases",
        "-c",
        default=None,
        help="Comma-separated case IDs to include (default: auto-discovery).",
    )
    parser.add_argument("--height", type=int, default=96, help="Output image height.")
    parser.add_argument("--width", type=int, default=96, help="Output image width.")
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "keep_ratio_pad", "keep_ratio_crop"],
        default="keep_ratio_pad",
        help="Resize strategy for tiny images.",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation.")
    parser.add_argument("--per-case-norm", action="store_true", help="Enable per-case pre normalization.")
    parser.add_argument("--global-norm", action="store_true", help="Enable global normalization (mean/std).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for preview loader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--preview", action="store_true", help="Print shapes of a preview batch.")
    parser.add_argument("--describe", action="store_true", help="Print dataset summary and exit.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    label_map = None
    if args.labels:
        label_map = load_label_mapping(args.labels)

    if args.cases:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_ids = None

    dataset = DCESequenceDataset(
        root_dir=args.root,
        case_ids=case_ids,
        label_map=label_map,
        image_size=(args.height, args.width),
        augment=not args.no_augment,
        per_case_pre_norm=args.per_case_norm,
        global_normalize=args.global_norm,
        return_dict=True,
        resize_mode=args.resize_mode,
    )

    if args.describe:
        print(dataset.describe())

    if args.preview:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dce_collate_fn,
        )
        batch = next(iter(loader))
        print("Batch keys:", batch.keys())
        print("images shape:", batch["images"].shape)
        print("mask shape:", batch["mask"].shape)
        if batch["label"] is not None:
            print("label shape:", batch["label"].shape)
        print("case IDs:", batch["case_id"])
        print("paths[0]:", batch["paths"][0])

    if not args.describe and not args.preview:
        print(dataset.describe())


if __name__ == "__main__":
    main()