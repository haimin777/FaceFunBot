#!/usr/bin/env python3
"""Funny photo compatibility based on lightweight image keypoints.

This is entertainment software: it compares visual signals in two photos and
turns them into a playful "couple verdict." It does not identify people or make
serious claims about relationships.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image, ImageOps


TARGET_SIZE = 384
MAX_KEYPOINTS = 180
DESCRIPTOR_RADIUS = 5
MatchMode = Literal["face", "palm"]


@dataclass(frozen=True)
class ImageFeatures:
    keypoints: list[tuple[int, int]]
    descriptors: np.ndarray
    brightness: float
    contrast: float
    warmth: float
    saturation: float
    center_of_energy: tuple[float, float]
    line_density: float
    line_direction: np.ndarray


@dataclass(frozen=True)
class MatchReport:
    mode: str
    score: int
    title: str
    verdict: str
    keypoint_matches: int
    keypoint_similarity: float
    color_harmony: float
    lighting_harmony: float
    composition_harmony: float
    line_harmony: float | None = None


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Photo not found: {path}")
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(TARGET_SIZE / image.width, TARGET_SIZE / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)

    average_color = tuple(int(channel) for channel in np.asarray(image).mean(axis=(0, 1)))
    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), average_color)
    offset = ((TARGET_SIZE - image.width) // 2, (TARGET_SIZE - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return array.copy()

    padded = np.pad(array, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = radius * 2 + 1
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / float(size * size)


def harris_response(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    ix2 = box_blur(gx * gx, 2)
    iy2 = box_blur(gy * gy, 2)
    ixy = box_blur(gx * gy, 2)
    det = ix2 * iy2 - ixy * ixy
    trace = ix2 + iy2
    return det - 0.04 * trace * trace


def local_maxima(response: np.ndarray, radius: int = 7) -> np.ndarray:
    mask = np.ones_like(response, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(response, dy, axis=0), dx, axis=1)
            mask &= response >= shifted
    margin = radius + DESCRIPTOR_RADIUS + 1
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def detect_keypoints(gray: np.ndarray) -> list[tuple[int, int]]:
    response = harris_response(gray)
    threshold = np.percentile(response, 98.8)
    candidates = np.argwhere(local_maxima(response) & (response > threshold))
    if candidates.size == 0:
        return []

    strengths = response[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(strengths)[::-1][:MAX_KEYPOINTS]
    return [(int(x), int(y)) for y, x in candidates[order]]


def build_descriptors(gray: np.ndarray, keypoints: Iterable[tuple[int, int]]) -> np.ndarray:
    descriptors = []
    r = DESCRIPTOR_RADIUS
    for x, y in keypoints:
        patch = gray[y - r : y + r + 1, x - r : x + r + 1].copy()
        patch -= patch.mean()
        norm = np.linalg.norm(patch)
        if norm < 1e-8:
            continue
        descriptors.append((patch / norm).reshape(-1))
    if not descriptors:
        return np.empty((0, (r * 2 + 1) ** 2), dtype=np.float32)
    return np.asarray(descriptors, dtype=np.float32)


def line_signature(gray: np.ndarray) -> tuple[float, np.ndarray]:
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    threshold = float(np.percentile(magnitude, 82))
    line_mask = magnitude > max(threshold, 1e-6)
    line_density = float(line_mask.mean())

    angles = (np.arctan2(gy, gx) + math.pi) % math.pi
    hist, _ = np.histogram(
        angles[line_mask],
        bins=8,
        range=(0.0, math.pi),
        weights=magnitude[line_mask],
    )
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-8:
        return line_density, np.zeros(8, dtype=np.float32)
    return line_density, (hist / norm).astype(np.float32)


def extract_features(path: Path) -> ImageFeatures:
    rgb = rgb_array(load_image(path))
    gray = grayscale(rgb)
    keypoints = detect_keypoints(gray)
    descriptors = build_descriptors(gray, keypoints)

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = float(np.mean(max_channel - min_channel))
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    warmth = float(np.mean(rgb[..., 0] - rgb[..., 2]) * 0.5 + 0.5)
    line_density, line_direction = line_signature(gray)

    energy = np.abs(gray - brightness)
    total_energy = float(energy.sum())
    if total_energy <= 1e-8:
        center = (0.5, 0.5)
    else:
        ys, xs = np.indices(gray.shape)
        center = (
            float((xs * energy).sum() / total_energy / (gray.shape[1] - 1)),
            float((ys * energy).sum() / total_energy / (gray.shape[0] - 1)),
        )

    return ImageFeatures(
        keypoints=keypoints,
        descriptors=descriptors,
        brightness=brightness,
        contrast=contrast,
        warmth=min(max(warmth, 0.0), 1.0),
        saturation=saturation,
        center_of_energy=center,
        line_density=line_density,
        line_direction=line_direction,
    )


def descriptor_matches(a: np.ndarray, b: np.ndarray) -> int:
    if len(a) == 0 or len(b) < 2:
        return 0

    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    nearest = np.partition(distances, kth=1, axis=1)[:, :2]
    good = nearest[:, 0] < nearest[:, 1] * 0.82
    return int(good.sum())


def closeness(left: float, right: float, tolerance: float) -> float:
    return max(0.0, 1.0 - abs(left - right) / tolerance)


def score_features(a: ImageFeatures, b: ImageFeatures, mode: MatchMode = "face") -> MatchReport:
    matches = descriptor_matches(a.descriptors, b.descriptors)
    keypoint_base = max(12, min(len(a.descriptors), len(b.descriptors)))
    keypoint_similarity = min(1.0, matches / keypoint_base)

    color_harmony = (
        closeness(a.warmth, b.warmth, 0.45) * 0.55
        + closeness(a.saturation, b.saturation, 0.35) * 0.45
    )
    lighting_harmony = (
        closeness(a.brightness, b.brightness, 0.5) * 0.65
        + closeness(a.contrast, b.contrast, 0.3) * 0.35
    )
    center_distance = math.dist(a.center_of_energy, b.center_of_energy)
    composition_harmony = max(0.0, 1.0 - center_distance / 0.65)
    line_harmony = palm_line_harmony(a, b)

    if mode == "palm":
        raw = (
            keypoint_similarity * 0.32
            + line_harmony * 0.30
            + color_harmony * 0.16
            + lighting_harmony * 0.12
            + composition_harmony * 0.10
        )
    else:
        raw = (
            keypoint_similarity * 0.45
            + color_harmony * 0.22
            + lighting_harmony * 0.18
            + composition_harmony * 0.15
        )

    # A tiny deterministic "chaos sparkle" makes the verdict feel less clinical.
    sparkle = ((matches * 17 + int(a.warmth * 100) + int(b.saturation * 100) + len(mode)) % 9) - 4
    score = int(round(min(100, max(0, raw * 100 + sparkle))))
    if mode == "palm":
        title, verdict = palm_verdict(score, keypoint_similarity, line_harmony, color_harmony)
    else:
        title, verdict = face_verdict(score, keypoint_similarity, color_harmony, lighting_harmony)

    return MatchReport(
        mode=mode,
        score=score,
        title=title,
        verdict=verdict,
        keypoint_matches=matches,
        keypoint_similarity=round(keypoint_similarity, 3),
        color_harmony=round(color_harmony, 3),
        lighting_harmony=round(lighting_harmony, 3),
        composition_harmony=round(composition_harmony, 3),
        line_harmony=round(line_harmony, 3) if mode == "palm" else None,
    )


def palm_line_harmony(a: ImageFeatures, b: ImageFeatures) -> float:
    direction_harmony = float(np.dot(a.line_direction, b.line_direction))
    density_harmony = closeness(a.line_density, b.line_density, 0.18)
    return max(0.0, min(1.0, direction_harmony * 0.7 + density_harmony * 0.3))


def face_verdict(
    score: int,
    keypoint_similarity: float,
    color_harmony: float,
    lighting_harmony: float,
) -> tuple[str, str]:
    if score >= 86:
        title = "Cosmic Main Characters"
    elif score >= 72:
        title = "Rom-Com With Good Lighting"
    elif score >= 58:
        title = "Chaotic But Photogenic"
    elif score >= 42:
        title = "Slow-Burn Side Quest"
    else:
        title = "Mismatched Album Covers"

    reasons = []
    if keypoint_similarity > 0.55:
        reasons.append("their keypoints are practically gossiping with each other")
    elif keypoint_similarity > 0.25:
        reasons.append("their faces share enough visual rhythm to start a group chat")
    else:
        reasons.append("their keypoints respectfully keep personal space")

    if color_harmony > 0.72:
        reasons.append("the color palette says coordinated without trying")
    elif color_harmony < 0.38:
        reasons.append("the colors are doing a bold opposites-attract routine")

    if lighting_harmony > 0.72:
        reasons.append("the lighting is suspiciously compatible")
    elif lighting_harmony < 0.38:
        reasons.append("one photo brought sunshine and the other brought plot tension")

    return title, f"{'; '.join(reasons)}."


def palm_verdict(
    score: int,
    keypoint_similarity: float,
    line_harmony: float,
    color_harmony: float,
) -> tuple[str, str]:
    if score >= 86:
        title = "Legendary Hand-Holding Forecast"
    elif score >= 72:
        title = "Palmistry Rom-Com Energy"
    elif score >= 58:
        title = "Cute Chaos In The Life Lines"
    elif score >= 42:
        title = "Potential With Dramatic Finger Guns"
    else:
        title = "Parallel Palms, Separate Side Quests"

    reasons = []
    if line_harmony > 0.72:
        reasons.append("their palm lines are suspiciously in sync")
    elif line_harmony > 0.45:
        reasons.append("their palm lines agree on the general plot but argue about details")
    else:
        reasons.append("their palm lines are each writing a very independent screenplay")

    if keypoint_similarity > 0.45:
        reasons.append("the little creases and landmarks keep finding each other")
    elif keypoint_similarity < 0.18:
        reasons.append("the keypoints are shy and require emotional snacks")

    if color_harmony > 0.72:
        reasons.append("the hand-photo vibes are warmly coordinated")
    elif color_harmony < 0.38:
        reasons.append("the color mood says opposites attract, possibly with jazz hands")

    return title, f"{'; '.join(reasons)}."


def compare_photos(first: Path, second: Path, mode: MatchMode = "face") -> MatchReport:
    return score_features(extract_features(first), extract_features(second), mode)


def format_report(report: MatchReport) -> str:
    lines = [
        f"{report.title}: {report.score}/100",
        report.verdict,
        "",
        f"Keypoint matches: {report.keypoint_matches}",
        f"Keypoint similarity: {report.keypoint_similarity:.3f}",
        f"Color harmony: {report.color_harmony:.3f}",
        f"Lighting harmony: {report.lighting_harmony:.3f}",
        f"Composition harmony: {report.composition_harmony:.3f}",
    ]
    if report.line_harmony is not None:
        lines.append(f"Palm line harmony: {report.line_harmony:.3f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two photos and produce a funny couple compatibility verdict."
    )
    parser.add_argument("first_photo", type=Path)
    parser.add_argument("second_photo", type=Path)
    parser.add_argument(
        "--mode",
        choices=("face", "palm"),
        default="face",
        help="Use face-style or palm-style compatibility text and scoring.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_photos(args.first_photo, args.second_photo, args.mode)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return

    print(format_report(report))


if __name__ == "__main__":
    main()
