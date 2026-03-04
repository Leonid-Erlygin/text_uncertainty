import shutil
import glob
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class ProtocolConfig:
    def __init__(self):
        # Paths
        self.base_dataset_dir = Path("/app/datasets/text_datasets")
        self.output_ident_dir = Path("/app/datasets/text-ident")
        self.output_ident_val_dir = Path("/app/datasets/text-ident-val")
        self.cache_features_dir = Path("/app/cache/features")

        # Training/Embedding Paths
        self.training_script_path = Path("/app/training/trainers/train.py")
        # self.checkpoint_path = Path("/app/outputs/text_scf/topic_dbpedia/last.ckpt")
        # Note: If you have specific checkpoints per dataset, use a dict:
        self.checkpoint_paths = {
            "yahoo": "/app/outputs/text_scf/topic_yahoo_answers/last.ckpt",
            "agnews": "/app/outputs/text_scf/topic_agnews/last.ckpt",
            "dbpedia": "/app/outputs/text_scf/topic_dbpedia/last.ckpt",
        }

        # Protocol Parameters
        self.template_idx_shift = 10000
        self.gallery_template_size_fraction = 1e-2
        self.min_gallery_template_size = 50
        self.probe_template_size = 1

        # Random State
        self.random_seed = 32

        # Datasets
        self.embedding_dataset_names = ["yahoo", "agnews", "dbpedia"]
        self.tokenizer_name = "bert-base-uncased"
        self.batch_size = 512
        self.max_length = 512
        self.num_workers = 32


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_distribution_data(
    protocol_path: Path, mode: str, match_test_ood_count: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads in-distribution and out-of-distribution data based on mode.

    Args:
        match_test_ood_count: If True (val mode), subsample OOD to match test set size
    """
    suffix = "test" if mode == "test" else "train"

    df_in = pd.read_csv(
        protocol_path / f"in_distribution_{suffix}.csv",
        header=None,
        usecols=[0, 2],
        names=["class", "text"],
        quotechar='"',
    )
    df_in["text"] = df_in["text"].str.strip("\"'")

    df_ood = pd.read_csv(
        protocol_path / f"out_distribution_{suffix}.csv",
        header=None,
        usecols=[0, 2],
        names=["class", "text"],
        quotechar='"',
    )
    df_ood["text"] = df_ood["text"].str.strip("\"'")

    # Subsample OOD for validation to match test set size
    if match_test_ood_count and mode == "val":
        df_ood_test = pd.read_csv(
            protocol_path / "out_distribution_test.csv",
            header=None,
            usecols=[0, 2],
            names=["class", "text"],
            quotechar='"',
        )
        test_ood_count = len(df_ood_test)
        if len(df_ood) > test_ood_count:
            ood_indices = np.random.default_rng(32).choice(
                len(df_ood), size=test_ood_count, replace=False
            )
            df_ood = df_ood.iloc[ood_indices].reset_index(drop=True)

    return df_in, df_ood


def save_texts_to_disk(save_dir: Path, texts: List[str], indices: List[int]) -> None:
    """
    Saves texts to disk using specific indices to maintain alignment with embeddings.
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    for idx, text in zip(indices, texts):
        file_path = save_dir / f"{idx}.txt"
        with open(file_path, "w", encoding="utf-8") as fd:
            fd.write(text)


# -----------------------------------------------------------------------------
# Protocol Construction Logic
# -----------------------------------------------------------------------------


def construct_splits(
    df_in: pd.DataFrame,
    df_ood: pd.DataFrame,
    df_in_ref: Optional[pd.DataFrame],
    config: ProtocolConfig,
    rng: np.random.Generator,
) -> Tuple[List[str], List[Any], List[int], List[str], List[Any], List[int]]:
    """
    Splits data into Gallery and Probe sets with template grouping.
    """
    known_classes, known_counts = np.unique(df_in["class"].values, return_counts=True)

    ref_counts = known_counts
    if df_in_ref is not None:
        ref_classes, ref_counts = np.unique(
            df_in_ref["class"].values, return_counts=True
        )
        class_count_map = dict(zip(ref_classes, ref_counts))
        ref_counts = np.array([class_count_map.get(c, 0) for c in known_classes])

    gallery_template_sizes = np.max(
        np.stack(
            [
                np.array([config.min_gallery_template_size] * len(known_classes)),
                (config.gallery_template_size_fraction * ref_counts).astype("int"),
            ],
            axis=1,
        ),
        axis=1,
    )

    gallery_paths = []
    gallery_sids = []
    gallery_tids = []

    probe_samples_buffer = []

    for i, known_class in enumerate(known_classes):
        class_mask = df_in["class"].values == known_class
        sample_indices = np.where(class_mask)[0]

        gallery_count = min(gallery_template_sizes[i], len(sample_indices))
        gallery_indices = rng.choice(sample_indices, size=gallery_count, replace=False)
        probe_indices = list(set(sample_indices) - set(gallery_indices))

        for idx in gallery_indices:
            gallery_paths.append(f"known_texts/{idx}.txt")
            gallery_sids.append(known_class)
            gallery_tids.append(known_class)

        for idx in probe_indices:
            probe_samples_buffer.append(
                {"path": f"known_texts/{idx}.txt", "sid": known_class, "orig_idx": idx}
            )

    ood_indices = np.arange(len(df_ood["text"].values))
    for idx in ood_indices:
        probe_samples_buffer.append(
            {
                "path": f"unknown_texts/{idx}.txt",
                "sid": df_ood["class"].values[idx],
                "orig_idx": idx,
            }
        )

    probe_paths = []
    probe_sids = []
    probe_tids = []

    probe_df = pd.DataFrame(probe_samples_buffer)

    if len(probe_df) == 0:
        return (
            gallery_paths,
            gallery_sids,
            gallery_tids,
            probe_paths,
            probe_sids,
            probe_tids,
        )

    unique_sids = probe_df["sid"].unique()
    current_tid = config.template_idx_shift

    for sid in unique_sids:
        sid_samples = probe_df[probe_df["sid"] == sid].to_dict("records")

        for i in range(0, len(sid_samples), config.probe_template_size):
            chunk = sid_samples[i : i + config.probe_template_size]

            for sample in chunk:
                probe_paths.append(sample["path"])
                probe_sids.append(sample["sid"])
                probe_tids.append(current_tid)

            current_tid += 1

    return (
        gallery_paths,
        gallery_sids,
        gallery_tids,
        probe_paths,
        probe_sids,
        probe_tids,
    )


def write_metadata(
    meta_path: Path,
    ds_name: str,
    gallery_paths: List[str],
    gallery_sids: List[Any],
    gallery_tids: List[int],
    probe_paths: List[str],
    probe_sids: List[Any],
    probe_tids: List[int],
) -> None:
    """
    Writes the metadata files (CSV and TXT) required for the protocol.
    """
    meta_path.mkdir(exist_ok=True, parents=True)

    all_paths = gallery_paths + probe_paths
    all_tids = gallery_tids + probe_tids
    all_sids = gallery_sids + probe_sids
    all_mids = list(range(len(all_paths)))

    tid_mid_file = meta_path / f"{ds_name}_face_tid_mid.txt"
    with open(tid_mid_file, "w", encoding="utf-8") as fd:
        for name, tid, sid, mid in zip(all_paths, all_tids, all_sids, all_mids):
            fd.write(f"{name} {tid} {mid} {sid}\n")

    probe_df = pd.DataFrame(
        {
            "TEMPLATE_ID": probe_tids,
            "SUBJECT_ID": probe_sids,
            "FILENAME": probe_paths,
        }
    )
    probe_file = meta_path / f"{ds_name}_1N_probe_mixed.csv"
    probe_df.to_csv(probe_file, sep=",", index=False)

    gallery_df = pd.DataFrame(
        {
            "TEMPLATE_ID": gallery_tids,
            "SUBJECT_ID": gallery_sids,
            "FILENAME": gallery_paths,
        }
    )
    gallery_file = meta_path / f"{ds_name}_1N_gallery_G1.csv"
    gallery_df.to_csv(gallery_file, sep=",", index=False)


def construct_protocol(
    protocol_path: Path,
    config: ProtocolConfig,
    rng: np.random.Generator,
    mode: str = "test",
) -> None:
    """
    Main function to construct a protocol (Test or Validation).
    """
    ds_name = protocol_path.stem.lower().split('_')[0]
    output_dir = (
        config.output_ident_dir if mode == "test" else config.output_ident_val_dir
    )

    print(f"Processing {mode} protocol: {ds_name}")

    # Load data with OOD subsampling for val mode
    df_in, df_ood = load_distribution_data(
        protocol_path,
        mode,
        match_test_ood_count=(mode == "val"),  # ← Enable subsampling for val
    )

    df_in_ref = None
    if mode == "val":
        df_in_ref, _ = load_distribution_data(protocol_path, "test")

    save_texts_to_disk(
        output_dir / ds_name / "known_texts", df_in["text"].values, df_in.index.tolist()
    )
    save_texts_to_disk(
        output_dir / ds_name / "unknown_texts",
        df_ood["text"].values,
        df_ood.index.tolist(),
    )

    (gallery_paths, gallery_sids, gallery_tids, probe_paths, probe_sids, probe_tids) = (
        construct_splits(df_in, df_ood, df_in_ref, config, rng)
    )

    meta_path = output_dir / ds_name / "meta"
    write_metadata(
        meta_path,
        ds_name,
        gallery_paths,
        gallery_sids,
        gallery_tids,
        probe_paths,
        probe_sids,
        probe_tids,
    )


# -----------------------------------------------------------------------------
# Embedding Computation
# -----------------------------------------------------------------------------


def compute_embeddings(config: ProtocolConfig) -> None:
    """
    Computes embeddings for all datasets and splits using the existing training infrastructure.
    """
    print("\n=== Computing Embeddings ===")

    if not config.training_script_path.exists():
        print(f"Warning: Training script not found at {config.training_script_path}")
        return

    splits = [
        {
            "mode": "val",
            "root_template": str(config.output_ident_val_dir / "{name}"),
            "output_suffix": "val",
        },
        {
            "mode": "test",
            "root_template": str(config.output_ident_dir / "{name}"),
            "output_suffix": "test",
        },
    ]

    for ds_name in config.embedding_dataset_names:
        if hasattr(config, "checkpoint_paths") and isinstance(
            config.checkpoint_paths, dict
        ):
            ckpt_path = config.checkpoint_paths.get(ds_name)
        else:
            ckpt_path = config.checkpoint_path

        if not Path(ckpt_path).exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}. Skipping {ds_name}.")
            continue

        for split in splits:
            root_dir = str(split["root_template"].format(name=ds_name))
            output_name = f"scf_2epoch_topic_{ds_name}_{split['output_suffix']}_embs"

            print(f"\nComputing embeddings for: {ds_name} ({split['mode']})")

            cmd = [
                "python3",
                str(config.training_script_path),
                f"-cn=text_model_{ds_name}_scf",
                "mode=predict",
                "~trainer.logger",
                f'+weights_path="{ckpt_path}"',
                # FIXED: Callback paths (index 1 = Prediction_writer)
                f'++trainer.callbacks.2.output_dir="{str(config.cache_features_dir)}"',
                f'++trainer.callbacks.2.file_name="{output_name}"',
                # FIXED: Predict dataset is under 'data', not 'trainer'
                f'++data.predict_dataset.root_dir="{root_dir}"',
                # Trainer params
                # f'+trainer.batch_size={config.batch_size}',
                # f'+trainer.max_length={config.max_length}',
                # f'+trainer.num_workers={config.num_workers}',
            ]

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✓ Success: {output_name}.npz")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {ds_name} ({split['mode']})")
                print(f"Error: {e.stderr}")


# -----------------------------------------------------------------------------
# Embedding Copy
# -----------------------------------------------------------------------------


def copy_embeddings(config: ProtocolConfig, mode: str) -> None:
    """
    Copies pre-computed embeddings from cache to the dataset directories.
    """
    if mode == "test":
        base_output_dir = config.output_ident_dir
        suffix = "test"
    else:
        base_output_dir = config.output_ident_val_dir
        suffix = "val"
    for name in config.embedding_dataset_names:
        embeddings_dir = base_output_dir / f"{name}" / "embeddings"
        embeddings_dir.mkdir(exist_ok=True, parents=True)

        src = config.cache_features_dir / f"scf_2epoch_topic_{name}_{suffix}_embs.npz"

        if src.is_file():
            shutil.copyfile(src, embeddings_dir / f"scf_embs_{name}.npz")
        else:
            print(f"Warning: Embedding file not found: {src}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main():
    config = ProtocolConfig()
    shutil.rmtree(config.output_ident_dir)
    shutil.rmtree(config.output_ident_val_dir)
    rng = np.random.default_rng(config.random_seed)

    # Ensure output directories exist
    config.output_ident_dir.mkdir(exist_ok=True, parents=True)
    config.output_ident_val_dir.mkdir(exist_ok=True, parents=True)
    config.cache_features_dir.mkdir(exist_ok=True, parents=True)

    # Find all protocol CSVs
    protocol_paths = list(glob.glob(str(config.base_dataset_dir / "*_csv")))

    if not protocol_paths:
        print(f"No protocols found in {config.base_dataset_dir}")
        return

    # 1. Construct Test Protocols
    print("\n=== Constructing Test Protocols ===")
    for protocol_path_str in protocol_paths:
        protocol_path = Path(protocol_path_str)
        construct_protocol(protocol_path, config, rng, mode="test")

    # 2. Construct Validation Protocols
    print("\n=== Constructing Validation Protocols ===")
    for protocol_path_str in protocol_paths:
        protocol_path = Path(protocol_path_str)
        construct_protocol(protocol_path, config, rng, mode="val")

    # 3. Compute Embeddings (NEW STEP)
    # print("\n=== Computing Embeddings ===")
    compute_embeddings(config)

    # 4. Copy Embeddings
    print("\n=== Copying Embeddings ===")
    copy_embeddings(config, mode="test")
    copy_embeddings(config, mode="val")

    print("\n=== Protocol Construction Complete ===")


if __name__ == "__main__":
    main()
