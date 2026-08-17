#!/usr/bin/env python3
"""Run the C++ t-SNE/explainer pipeline on a headered CSV and create plots."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run t-SNE explanations on a CSV dataset and visualize them."
    )
    parser.add_argument("--input", required=True, type=Path, help="Headered CSV dataset")
    parser.add_argument("--label-column", help="Optional column used to color the embedding")
    parser.add_argument(
        "--delimiter",
        help="Column delimiter, for example ',' or ';' (default: detect automatically)",
    )
    parser.add_argument("--perplexity", type=float, help="Default: chosen from the sample count")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0, help="Sample row to explain")
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--binary-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path)
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        help="Standard-scale features to zero mean and unit variance (default)",
    )
    scaling.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Use raw numeric values without standard scaling",
    )
    parser.set_defaults(standardize=True)
    return parser.parse_args()


def find_binary(directory: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    filename = name + suffix
    candidates = [
        directory / filename,
        directory / "Release" / filename,
        directory / "release" / filename,
        directory / "Debug" / filename,
        directory / "debug" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {filename}. Checked:\n  {checked}")


def load_features(
    input_path: Path,
    label_column: str | None,
    standardize: bool,
    delimiter: str | None,
) -> tuple[pd.DataFrame, pd.Series | None, np.ndarray, list[str]]:
    frame = pd.read_csv(input_path, sep=delimiter, engine="python" if delimiter is None else "c")
    if frame.empty:
        raise ValueError("The input CSV contains no data rows")

    labels = None
    if label_column:
        if label_column not in frame.columns:
            raise ValueError(f"Label column {label_column!r} was not found")
        labels = frame[label_column].copy()
        frame = frame.drop(columns=[label_column])

    non_numeric = frame.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        names = ", ".join(map(str, non_numeric))
        raise ValueError(
            f"Non-numeric feature columns: {names}. Remove them or select one with --label-column."
        )
    if frame.shape[1] == 0:
        raise ValueError("No numeric feature columns remain")
    if frame.shape[0] < 3:
        raise ValueError("At least three samples are required")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain missing or non-finite values")

    feature_names = [str(column) for column in frame.columns]
    values = frame.to_numpy(dtype=float)
    if standardize:
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        constant = scales == 0
        if constant.any():
            names = ", ".join(np.asarray(feature_names)[constant])
            print(f"Warning: constant features kept with zero values after scaling: {names}")
            scales[constant] = 1.0
        values = (values - means) / scales
    return frame, labels, values, feature_names


def choose_perplexity(requested: float | None, sample_count: int) -> float:
    value = requested
    if value is None:
        value = min(30.0, max(2.0, (sample_count - 1) / 3.0))
        print(f"Using automatic perplexity: {value:g}")
    if not 0 < value < sample_count:
        raise ValueError(
            f"Perplexity must be positive and smaller than {sample_count}; got {value}"
        )
    return value


def save_embedding_plot(
    embedding: np.ndarray,
    labels: pd.Series | None,
    sample: int,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    if labels is None:
        axis.scatter(embedding[:, 0], embedding[:, 1], s=35, alpha=0.8)
    else:
        text_labels = labels.fillna("<missing>").astype(str).to_numpy()
        categories = pd.unique(text_labels)
        if len(categories) <= 20:
            colors = plt.get_cmap("tab20", max(len(categories), 1))
            for index, category in enumerate(categories):
                mask = text_labels == category
                axis.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=35,
                    alpha=0.8,
                    color=colors(index),
                    label=category,
                )
            axis.legend(title=labels.name, bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            codes = pd.Categorical(text_labels).codes
            axis.scatter(embedding[:, 0], embedding[:, 1], c=codes, cmap="tab20", s=35)
    axis.scatter(
        embedding[sample, 0],
        embedding[sample, 1],
        s=130,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
    )
    axis.annotate(str(sample), embedding[sample], xytext=(5, 5), textcoords="offset points")
    axis.set(title="t-SNE embedding", xlabel="t-SNE axis 1", ylabel="t-SNE axis 2")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_importance_plot(
    values: np.ndarray,
    feature_names: list[str],
    title: str,
    path: Path,
    top: int,
    xlabel: str = "Gradient magnitude",
) -> None:
    count = min(top, len(feature_names))
    selected = np.argsort(values)[-count:]
    fig_height = max(4.0, 0.35 * count + 1.5)
    fig, axis = plt.subplots(figsize=(8, fig_height))
    axis.barh(np.asarray(feature_names)[selected], values[selected])
    axis.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def paper_global_importance(importance: np.ndarray) -> np.ndarray:
    """Equation (9): mean per-instance normalized gradient magnitude."""
    totals = importance.sum(axis=1, keepdims=True)
    normalized = np.divide(
        importance,
        totals,
        out=np.zeros_like(importance),
        where=totals > 0,
    )
    return normalized.mean(axis=0)


def save_axis_gradient_plot(
    values: np.ndarray,
    feature_names: list[str],
    axis_number: int,
    sample: int,
    path: Path,
    top: int,
) -> None:
    count = min(top, len(feature_names))
    selected = np.argsort(np.abs(values))[-count:]
    selected_values = values[selected]
    colors = np.where(selected_values >= 0, "#2878B5", "#D9534F")
    fig_height = max(4.0, 0.35 * count + 1.5)
    fig, axis = plt.subplots(figsize=(8, fig_height))
    axis.barh(np.asarray(feature_names)[selected], selected_values, color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"Sample {sample}: explanation along t-SNE axis {axis_number}",
        xlabel="Signed gradient",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.top_features <= 0:
        raise ValueError("--top-features must be positive")

    _, labels, values, feature_names = load_features(
        args.input, args.label_column, args.standardize, args.delimiter
    )
    sample_count, feature_count = values.shape
    if not 0 <= args.sample < sample_count:
        raise ValueError(f"--sample must be between 0 and {sample_count - 1}")
    perplexity = choose_perplexity(args.perplexity, sample_count)

    output_dir = args.output_dir or Path("results") / args.input.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    tsne_binary = find_binary(args.binary_dir, "tsne")
    explainer_binary = find_binary(args.binary_dir, "tsne_explainer")

    numeric_input = output_dir / "input_numeric.csv"
    prefix = output_dir / "tsne"
    raw_gradients_path = output_dir / "raw_gradients.csv"
    np.savetxt(numeric_input, values, delimiter=",", fmt="%.17g")

    subprocess.run(
        [
            str(tsne_binary),
            str(numeric_input),
            str(perplexity),
            str(prefix),
            "--iterations",
            str(args.iterations),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(explainer_binary),
            str(numeric_input),
            str(prefix) + "_Y.csv",
            str(prefix) + "_P.csv",
            str(prefix) + "_Q.csv",
            str(prefix) + "_sigma.csv",
            str(raw_gradients_path),
        ],
        check=True,
    )

    embedding = np.atleast_2d(np.loadtxt(str(prefix) + "_Y.csv", delimiter=","))
    raw_gradients = np.atleast_2d(np.loadtxt(raw_gradients_path, delimiter=","))
    expected_shape = (sample_count, 2 * feature_count)
    if raw_gradients.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected gradient shape {raw_gradients.shape}; expected {expected_shape}"
        )

    gradient_x = raw_gradients[:, :feature_count]
    gradient_y = raw_gradients[:, feature_count:]
    importance = np.hypot(gradient_x, gradient_y)

    embedding_frame = pd.DataFrame(embedding, columns=["tsne_x", "tsne_y"])
    embedding_frame.insert(0, "sample_index", np.arange(sample_count))
    if labels is not None:
        embedding_frame[args.label_column] = labels.to_numpy()
    embedding_frame.to_csv(output_dir / "embedding.csv", index=False)

    gradient_data: dict[str, np.ndarray] = {"sample_index": np.arange(sample_count)}
    for index, feature in enumerate(feature_names):
        gradient_data[f"axis_1__{feature}"] = gradient_x[:, index]
        gradient_data[f"axis_2__{feature}"] = gradient_y[:, index]
        gradient_data[f"importance__{feature}"] = importance[:, index]
    pd.DataFrame(gradient_data).to_csv(output_dir / "gradients.csv", index=False)

    save_embedding_plot(embedding, labels, args.sample, output_dir / "embedding.png")
    save_importance_plot(
        paper_global_importance(importance),
        feature_names,
        "Global feature importance (paper Eq. 9)",
        output_dir / "global_feature_importance.png",
        args.top_features,
        "Mean normalized importance",
    )
    save_importance_plot(
        np.abs(gradient_x).mean(axis=0),
        feature_names,
        "Global mean absolute gradient along t-SNE axis 1",
        output_dir / "global_mean_absolute_gradient_axis_1.png",
        args.top_features,
        "Mean absolute axis-1 gradient",
    )
    save_importance_plot(
        np.abs(gradient_y).mean(axis=0),
        feature_names,
        "Global mean absolute gradient along t-SNE axis 2",
        output_dir / "global_mean_absolute_gradient_axis_2.png",
        args.top_features,
        "Mean absolute axis-2 gradient",
    )
    save_importance_plot(
        importance[args.sample],
        feature_names,
        f"Feature importance for sample {args.sample}",
        output_dir / f"sample_{args.sample}_explanation.png",
        args.top_features,
    )
    save_axis_gradient_plot(
        gradient_x[args.sample],
        feature_names,
        1,
        args.sample,
        output_dir / f"sample_{args.sample}_axis_1_explanation.png",
        args.top_features,
    )
    save_axis_gradient_plot(
        gradient_y[args.sample],
        feature_names,
        2,
        args.sample,
        output_dir / f"sample_{args.sample}_axis_2_explanation.png",
        args.top_features,
    )

    print(f"Completed {sample_count} samples with {feature_count} features.")
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
