"""Reproducible processed-data snapshots for the eight IJAR case studies.

Writes exactly the post-preprocessing predictor matrix, target and ordered feature
names used by the main experiments.  No raw, third-party dataset is modified.

Only publish exported Parquet files if permitted by their underlying data licences.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd


SOURCES = {
    "diabetes": {
        "provider": "scikit-learn",
        "dataset": "load_diabetes",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "target": "target",
    },
    "engel": {
        "provider": "statsmodels",
        "dataset": "statsmodels.datasets.engel",
        "documentation": "https://www.statsmodels.org/stable/datasets/generated/engel.html",
        "target": "foodexp",
    },
    "concrete": {
        "provider": "OpenML", "data_id": 44959,
        "documentation": "https://www.openml.org/d/44959", "target": "strength",
    },
    "abalone": {
        "provider": "OpenML", "data_id": 183,
        "documentation": "https://www.openml.org/d/183", "target": "Class_number_of_rings",
    },
    "red_wine": {
        "provider": "OpenML", "data_id": 42184,
        "documentation": "https://www.openml.org/d/42184", "target": "quality",
    },
    "white_wine": {
        "provider": "OpenML", "data_id": 40498,
        "documentation": "https://www.openml.org/d/40498", "target": "Class",
    },
    "auto_mpg": {
        "provider": "Rdatasets / ISLR", "dataset": "Auto",
        "documentation": "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Auto.csv",
        "target": "mpg",
    },
    "windsor": {
        "provider": "Rdatasets / Ecdat", "dataset": "Housing",
        "documentation": "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Housing.csv",
        "target": "price",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "pyarrow"):
        try:
            pkg = __import__(module)
            info[module] = pkg.__version__
        except (ImportError, AttributeError):
            info[module] = None
    return info


def export_processed_datasets(specs, destination: Path, overwrite: bool = False) -> Path:
    """Save processed arrays to Parquet and record metadata/checksums.

    snapshots are NOT train/validation/test partitions: splits depend on
    the RNG seeds in real_data_experiments.py.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"{destination / 'manifest.json'} already exists; use --overwrite intentionally")
    output = {"schema_version": 1,
              "description": "Post-preprocessing covariates and response, before train/validation/test splitting.",
              "not_raw_source_data": True,
              "processing": "Exact DatasetSpec arrays from real_data_experiments.load_datasets.",
              "environment": environment_info(),
              "datasets": []}
    for spec in specs:
        if spec.name not in SOURCES:
            raise ValueError(f"Unexpected dataset: {spec.name}")
        X = np.asarray(spec.X, dtype=np.float64)
        y = np.asarray(spec.y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"Invalid shapes for {spec.name}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite data in {spec.name}")
        feature_names = list(spec.feature_names)
        if len(set(feature_names)) != len(feature_names) or "__target__" in feature_names:
            raise ValueError(f"Duplicate or reserved feature name in {spec.name}")
        frame = pd.DataFrame(X, columns=feature_names)
        frame["__target__"] = y
        filename = f"{spec.name}.parquet"
        outfile = destination / filename
        if outfile.exists() and not overwrite:
            raise FileExistsError(f"{outfile} exists; use --overwrite intentionally")
        frame.to_parquet(outfile, index=False, engine="pyarrow")
        output["datasets"].append({
            "name": spec.name,
            "file": filename,
            "sha256": file_sha256(outfile),
            "n_observations": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": feature_names,
            "directions": [int(v) for v in spec.directions],
            "min_leaf": int(spec.min_leaf),
            "source": SOURCES[spec.name],
        })
        print(f"Exported {filename}: n={X.shape[0]}, p={X.shape[1]}")
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_processed_datasets(destination: Path):
    """Verify checksums and reconstruct the original DatasetSpec objects.

    All experiments can then run from fixed exported data without fetching
    potentially updated third-party datasets.
    """
    # Import on demand to avoid a circular dependency at module load time.
    from real_data_experiments import DatasetSpec

    destination = Path(destination)
    metadata = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unknown manifest schema version")
    rows = metadata.get("datasets", [])
    if {d["name"] for d in rows} != set(SOURCES):
        raise ValueError("The snapshot must contain all eight named datasets")
    specs = []
    for entry in rows:
        name = entry["name"]
        fpath = destination / f"{name}.parquet"
        if fpath.name != entry["file"] or not fpath.is_file():
            raise FileNotFoundError(f"Missing or renamed parquet file for {name}")
        actual = file_sha256(fpath)
        if actual != entry["sha256"]:
            raise ValueError(f"Checksum mismatch in {fpath}; snapshot was changed")
        data = pd.read_parquet(fpath, engine="pyarrow")
        names = tuple(entry["feature_names"])
        if tuple(data.columns) != names + ("__target__",):
            raise ValueError(f"Columns or order changed for {name}")
        X = data.loc[:, list(names)].to_numpy(dtype=np.float64)
        y = data["__target__"].to_numpy(dtype=np.float64)
        directions = np.asarray(entry["directions"], dtype=int)
        if (X.shape != (entry["n_observations"], entry["n_features"])
                or directions.shape != (X.shape[1],)):
            raise ValueError(f"Shape mismatch in {name}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite values in {name}")
        specs.append(DatasetSpec(name=name, X=X, y=y,
                                 feature_names=names, directions=directions,
                                 min_leaf=int(entry["min_leaf"])))
    return specs
