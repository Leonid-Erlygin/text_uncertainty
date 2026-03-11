import shutil
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import subprocess
import glob


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class ProtocolConfig:
    def __init__(self):
        # Paths
        self.cache_features_dir = Path("/app/cache/features")
        self.training_script_path = Path("/app/training/trainers/train.py")

        # CLINC150 Paths
        self.clinc150_data_path = Path("/app/datasets/clinc150/data_full.json")
        self.clinc150_val_dir = Path("/app/outputs/clinc150_val")
        self.clinc150_test_dir = Path("/app/outputs/clinc150_test")

        # self.clinc150_val_dir = Path("/app/datasets/clinc150_val")
        # self.clinc150_test_dir = Path("/app/datasets/clinc150_test")

        # PAN Paths
        self.pan_train_jsonl = Path(
            "/app/datasets/pan/unseen_authors/xl/pan20-av-large-notest.jsonl"
        )
        self.pan_test_jsonl = Path(
            "/app/datasets/pan/unseen_authors/xl/pan20-av-large-test.jsonl"
        )
        # self.pan_val_dir = Path("/app/datasets/pan_val")
        # self.pan_test_dir = Path("/app/datasets/pan_test")
        self.pan_val_dir = Path("/app/outputs/pan_val")
        self.pan_test_dir = Path("/app/outputs/pan_test")

        # Protocol Parameters
        self.template_idx_shift = 10000
        self.gallery_docs_per_author = 3

        # Random State
        self.random_seed = 32

        # Dataset Settings
        self.tokenizer_name = "bert-base-uncased"
        self.batch_size = 64
        self.max_length = 512
        self.num_workers = 16

        # Checkpoints
        self.checkpoint_paths = {
            "clinc150": "/app/model_weights/text_models/trained_scf/clinc150.ckpt",
            "pan": "/app/model_weights/text_models/trained_scf/pan.ckpt",
        }


# -----------------------------------------------------------------------------
# CLINC150 Protocol Construction
# -----------------------------------------------------------------------------
def build_clinc150_protocol(
    datamodule,
    output_dir: Path,
    ds_type: str = "val",
    template_idx_shift: int = 10000,
):
    """
    Construct CLINC150 protocol with gallery/probe splits.
    Order: [train_texts (gallery)] + [val/test_texts (probe)]
    """
    rng = np.random.default_rng(32)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create directories
    text_save_dir = output_dir / "texts"
    text_save_dir.mkdir(exist_ok=True)
    meta_path = output_dir / "meta"
    meta_path.mkdir(exist_ok=True, parents=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Initialize dataset
    if ds_type == "val":
        ds = datamodule.val_dataset
        name_appendix = "_val"
    elif ds_type == "test":
        ds = datamodule.test_dataset
        name_appendix = ""
    else:
        raise ValueError(f"Unknown ds_type: {ds_type}")

    # Save texts into files
    text_counter = 0
    gallery_paths = []
    gallery_ids = []
    probe_paths = []
    probe_ids = []

    # First save train texts (gallery)
    for i in range(len(datamodule.train_dataset)):
        gallery_ids.append(datamodule.train_dataset[i]["label"])
        gallery_paths.append(f"texts/{text_counter}.txt")
        with open(text_save_dir / f"{text_counter}.txt", "w") as fd:
            fd.write(datamodule.train_dataset[i]["text"])
        text_counter += 1

    # Then save val/test texts (probe)
    for i in range(len(ds)):
        probe_ids.append(ds[i]["label"])
        probe_paths.append(f"texts/{text_counter}.txt")
        with open(text_save_dir / f"{text_counter}.txt", "w") as fd:
            fd.write(ds[i]["text"])
        text_counter += 1

    # Convert to arrays
    gallery_paths = np.array(gallery_paths)
    gallery_ids = np.array(gallery_ids)
    probe_paths = np.array(probe_paths)
    probe_ids = np.array(probe_ids)
    probe_template_ids = np.arange(len(probe_ids)) + template_idx_shift

    # Create tid/mid file
    text_paths = np.concatenate([gallery_paths, probe_paths])
    ids = np.concatenate([gallery_ids, probe_ids])
    tids = np.concatenate([gallery_ids, probe_template_ids])
    mids = np.arange(len(ids))

    out_file_tid_mid = meta_path / Path(f"clinc150{name_appendix}_face_tid_mid.txt")
    with open(out_file_tid_mid, "w") as fd:
        for name, tid, mid, sid in zip(text_paths, tids, ids, mids):
            fd.write(f"{name} {tid} {mid} {sid}\n")

    # Create gallery and probe meta files
    out_file_probe = meta_path / Path(f"clinc150{name_appendix}_1N_probe_mixed.csv")
    out_file_gallery = meta_path / Path(f"clinc150{name_appendix}_1N_gallery_G1.csv")

    assert len(gallery_ids) + len(probe_ids) == len(text_paths)

    probe = pd.DataFrame(
        {
            "TEMPLATE_ID": probe_template_ids,
            "SUBJECT_ID": probe_ids,
            "FILENAME": probe_paths,
        }
    )
    gallery = pd.DataFrame(
        {
            "TEMPLATE_ID": gallery_ids,
            "SUBJECT_ID": gallery_ids,
            "FILENAME": gallery_paths,
        }
    )

    probe.to_csv(out_file_probe, sep=",", index=False)
    gallery.to_csv(out_file_gallery, sep=",", index=False)

    return gallery_paths, probe_paths


# -----------------------------------------------------------------------------
# PAN Protocol Construction
# -----------------------------------------------------------------------------
def build_pan_protocol_exact_order(
    datamodule,
    output_dir: Path,
    ds_type: str = "val",
    gallery_docs_per_author: int = 3,
    template_idx_shift: int = 10000,
    ds_name: str = "pan",
    use_one_sample_per_template: bool = False,
):
    """
    Construct PAN OSR protocol with STRICT index alignment:
    Line N in tid_mid.txt ↔ texts/N.txt ↔ embeddings.npz[N]
    Order: [gallery_samples (3 per author)] + [probe_samples (val_dataset order)]
    """
    rng = np.random.default_rng(0)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create directories
    text_save_dir = output_dir / "texts"
    text_save_dir.mkdir(exist_ok=True)
    meta_path = output_dir / "meta"
    meta_path.mkdir(exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Step 1: Determine source JSONL and author lists based on ds_type
    if ds_type == "val":
        gallery_authors = datamodule.val_in_gallery
        out_of_gallery_authors = datamodule.val_out_gallery
        dataset = datamodule.val_dataset
    elif ds_type == "test":
        gallery_authors = datamodule.test_in_gallery
        out_of_gallery_authors = datamodule.test_out_gallery
        dataset = datamodule.test_dataset
    else:
        raise ValueError(f"Unknown ds_type: {ds_type}")

    index_to_meta = defaultdict(list)
    author_to_ids = defaultdict(list)

    for i in range(len(dataset)):
        author_to_ids[dataset[i]["author_id"]].append(i)
        # Create text entry
        with open(text_save_dir / f"{i}.txt", "w") as fd:
            fd.write(dataset[i]["text"])

    author_id = 0
    media_id = 0
    probe_template_id = 0
    probe_template_ids = []
    probe_ids = []
    probe_paths = []
    gallery_paths = []
    gallery_ids = []

    # Process gallery authors
    for gallery_author in gallery_authors:
        in_gallery_samples = rng.choice(
            author_to_ids[gallery_author], gallery_docs_per_author, replace=False
        )

        for in_gallery_sample in in_gallery_samples:
            name = f"texts/{in_gallery_sample}.txt"
            index_to_meta[in_gallery_sample] = [name, author_id, media_id, author_id]
            gallery_ids.append(author_id)
            gallery_paths.append(name)
            media_id += 1

        # Remaining samples from gallery authors go to probe
        for probe_in_gallery_sample in set(author_to_ids[gallery_author]) - set(
            in_gallery_samples
        ):
            name = f"texts/{probe_in_gallery_sample}.txt"
            index_to_meta[probe_in_gallery_sample] = [
                name,
                probe_template_id + template_idx_shift,
                media_id,
                author_id,
            ]
            probe_template_ids.append(probe_template_id + template_idx_shift)
            probe_ids.append(author_id)
            probe_paths.append(name)
            media_id += 1

            if use_one_sample_per_template:
                probe_template_id += 1

        if not use_one_sample_per_template:
            probe_template_id += 1
        author_id += 1

    # Process out-of-gallery authors
    for out_of_gallery_author in out_of_gallery_authors:
        for oog_sample_id in author_to_ids[out_of_gallery_author]:
            name = f"texts/{oog_sample_id}.txt"
            index_to_meta[oog_sample_id] = [
                name,
                probe_template_id + template_idx_shift,
                media_id,
                author_id,
            ]
            probe_template_ids.append(probe_template_id + template_idx_shift)
            probe_ids.append(author_id)
            probe_paths.append(name)

            if use_one_sample_per_template:
                probe_template_id += 1
            media_id += 1
            author_id += 1

        if not use_one_sample_per_template:
            probe_template_id += 1

    # Write tid_mid file
    out_file_tid_mid = meta_path / Path(f"{ds_name}_{ds_type}_face_tid_mid.txt")
    with open(out_file_tid_mid, "w") as fd:
        for i in range(len(dataset)):
            name, tid, mid, sid = index_to_meta[i]
            fd.write(f"{name} {tid} {mid} {sid}\n")

    # Create gallery and probe meta files
    out_file_probe = meta_path / Path(f"{ds_name}_{ds_type}_1N_probe_mixed.csv")
    out_file_gallery = meta_path / Path(f"{ds_name}_{ds_type}_1N_gallery_G1.csv")

    assert len(gallery_ids) + len(probe_ids) == len(dataset)

    probe = pd.DataFrame(
        {
            "TEMPLATE_ID": probe_template_ids,
            "SUBJECT_ID": probe_ids,
            "FILENAME": probe_paths,
        }
    )
    gallery = pd.DataFrame(
        {
            "TEMPLATE_ID": gallery_ids,
            "SUBJECT_ID": gallery_ids,
            "FILENAME": gallery_paths,
        }
    )

    probe.to_csv(out_file_probe, sep=",", index=False)
    gallery.to_csv(out_file_gallery, sep=",", index=False)

    return gallery_paths, probe_paths


# -----------------------------------------------------------------------------
# Embedding Management
# -----------------------------------------------------------------------------
def copy_clinc150_embeddings(config: ProtocolConfig, ds_type: str):
    """Copy CLINC150 embeddings from cache to dataset directory."""
    if ds_type == "val":
        output_dir = config.clinc150_val_dir
        src_emb = config.cache_features_dir / "clinc150_val_embs.npz"
        train_emb = config.cache_features_dir / "clinc150_train_embs.npz"
        dst_emb = output_dir / "embeddings" / "scf_embs_clinc150_val.npz"
    elif ds_type == "test":
        output_dir = config.clinc150_test_dir
        src_emb = config.cache_features_dir / "clinc150_test_embs.npz"
        train_emb = config.cache_features_dir / "clinc150_train_embs.npz"
        dst_emb = output_dir / "embeddings" / "scf_embs_clinc150.npz"
    else:
        raise ValueError(f"Unknown ds_type: {ds_type}")

    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "embeddings").mkdir(exist_ok=True)

    # Load and concatenate train + val/test embeddings
    train_embs = np.load(train_emb)
    val_embs = np.load(src_emb)

    np.savez(
        dst_emb,
        embs=np.concatenate([train_embs["embs"], val_embs["embs"]], axis=0),
        unc=np.concatenate([train_embs["unc"], val_embs["unc"]], axis=0),
    )
    print(f"✓ CLINC150 {ds_type} embeddings saved to {dst_emb}")


def copy_pan_embeddings(config: ProtocolConfig, ds_type: str):
    """Copy PAN embeddings from cache to dataset directory."""
    if ds_type == "val":
        output_dir = config.pan_val_dir
        src_emb = config.cache_features_dir / "pan_val_embs.npz"
        dst_emb = output_dir / "embeddings" / "scf_embs_pan_val.npz"
    elif ds_type == "test":
        output_dir = config.pan_test_dir
        src_emb = config.cache_features_dir / "pan_test_embs.npz"
        dst_emb = output_dir / "embeddings" / "scf_embs_pan_test.npz"
    else:
        raise ValueError(f"Unknown ds_type: {ds_type}")

    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "embeddings").mkdir(exist_ok=True)

    shutil.copyfile(src_emb, dst_emb)
    print(f"✓ PAN {ds_type} embeddings saved to {dst_emb}")


def compute_embeddings(config: ProtocolConfig, dataset_name: str, ds_type: str):
    """Compute embeddings using training script."""
    if ds_type == "val":
        if dataset_name == "clinc150":
            root_dir = str(config.clinc150_val_dir)
            output_name = "scf_embs_clinc150_val"
        else:  # pan
            root_dir = str(config.pan_val_dir)
            output_name = "scf_embs_pan_val"
    else:  # test
        if dataset_name == "clinc150":
            root_dir = str(config.clinc150_test_dir)
            output_name = "scf_embs_clinc150"
        else:  # pan
            root_dir = str(config.pan_test_dir)
            output_name = "scf_embs_pan_test"

    ckpt_path = config.checkpoint_paths.get(dataset_name)
    if not ckpt_path or not Path(ckpt_path).exists():
        print(f"Warning: Checkpoint not found: {ckpt_path}. Skipping {dataset_name}.")
        return

    cmd = [
        "python3",
        str(config.training_script_path),
        f"-cn=text_model_{dataset_name}_scf",
        "mode=predict",
        "~trainer.logger",
        f'+weights_path="{ckpt_path}"',
        f'++trainer.callbacks.2.output_dir="{str(config.cache_features_dir)}"',
        f'++trainer.callbacks.2.file_name="{output_name}"',
        # f'++data.predict_dataset.root_dir="{root_dir}"',
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Success: {output_name}.npz")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {dataset_name} ({ds_type})")
        print(f"Error: {e.stderr}")


# -----------------------------------------------------------------------------
# Main Protocol Construction Functions
# -----------------------------------------------------------------------------
def construct_clinc150_protocol(config: ProtocolConfig, ds_type: str = "val"):
    """Construct full CLINC150 protocol (texts + metadata + embeddings)."""
    from training.dataset_classes.text_datasets import Clinc150DataModule

    print(f"\n=== Constructing CLINC150 {ds_type.upper()} Protocol ===")

    # Initialize datamodule
    datamodule = Clinc150DataModule(str(config.clinc150_data_path))
    datamodule.setup()

    # Set output directory
    if ds_type == "val":
        output_dir = config.clinc150_val_dir
    else:
        output_dir = config.clinc150_test_dir

    # Clean up old files if they exist
    if output_dir.exists():
        meta_path = output_dir / "meta"
        if meta_path.exists():
            for f in meta_path.glob("*.txt"):
                f.unlink()
            for f in meta_path.glob("*.csv"):
                f.unlink()

    # Build protocol
    build_clinc150_protocol(
        datamodule=datamodule,
        output_dir=output_dir,
        ds_type=ds_type,
        template_idx_shift=config.template_idx_shift,
    )

    # Copy embeddings
    copy_clinc150_embeddings(config, ds_type)

    print(f"✓ CLINC150 {ds_type} protocol construction complete")


def construct_pan_protocol(config: ProtocolConfig, ds_type: str = "val"):
    """Construct full PAN protocol (texts + metadata + embeddings)."""
    from training.dataset_classes.pan_text import PANDataModule

    print(f"\n=== Constructing PAN {ds_type.upper()} Protocol ===")

    # Initialize datamodule
    dm = PANDataModule(
        train_jsonl=str(config.pan_train_jsonl),
        test_jsonl=str(config.pan_test_jsonl),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        tokenizer_name=config.tokenizer_name,
        max_length=config.max_length,
        min_docs_per_author=10,
        train_authors=4000,
        val_authors=200,
        val_probe_authors=200,
        test_authors=200,
        test_probe_authors=200,
    )
    dm.setup()

    # Set output directory
    if ds_type == "val":
        output_dir = config.pan_val_dir
    else:
        output_dir = config.pan_test_dir

    # Clean up old files if they exist
    if output_dir.exists():
        backup_files = ["backup.npz", "gallery_prob_backup.npz"]
        for bf in backup_files:
            bf_path = output_dir / bf
            if bf_path.exists():
                bf_path.unlink()
        meta_path = output_dir / "meta"
        if meta_path.exists():
            for f in meta_path.glob("*.txt"):
                f.unlink()
            for f in meta_path.glob("*.csv"):
                f.unlink()

    # Build protocol
    build_pan_protocol_exact_order(
        datamodule=dm,
        output_dir=output_dir,
        ds_type=ds_type,
        gallery_docs_per_author=config.gallery_docs_per_author,
        template_idx_shift=config.template_idx_shift,
        use_one_sample_per_template=False,
    )

    # Copy embeddings
    copy_pan_embeddings(config, ds_type)

    print(f"✓ PAN {ds_type} protocol construction complete")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    config = ProtocolConfig()
    rng = np.random.default_rng(config.random_seed)

    # Ensure directories exist
    config.cache_features_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("PAN & CLINC150 Protocol Construction")
    print("=" * 60)

    # =========================================================
    # STEP 1: Compute embeddings FIRST (before protocol construction)
    # =========================================================
    print("\n=== Step 1: Computing Embeddings ===")
    compute_embeddings(config, dataset_name="clinc150", ds_type="val")
    compute_embeddings(config, dataset_name="clinc150", ds_type="test")
    compute_embeddings(config, dataset_name="pan", ds_type="val")
    compute_embeddings(config, dataset_name="pan", ds_type="test")

    # =========================================================
    # STEP 2: Construct CLINC150 protocols
    # =========================================================
    print("\n=== Step 2: Constructing CLINC150 Protocols ===")
    construct_clinc150_protocol(config, ds_type="val")
    construct_clinc150_protocol(config, ds_type="test")

    # =========================================================
    # STEP 3: Construct PAN protocols
    # =========================================================
    print("\n=== Step 3: Constructing PAN Protocols ===")
    construct_pan_protocol(config, ds_type="val")
    construct_pan_protocol(config, ds_type="test")

    print("\n" + "=" * 60)
    print("Protocol Construction Complete")
    print("=" * 60)
    print("\nOutput directories:")
    print(f"  CLINC150 Val:  {config.clinc150_val_dir}")
    print(f"  CLINC150 Test: {config.clinc150_test_dir}")
    print(f"  PAN Val:       {config.pan_val_dir}")
    print(f"  PAN Test:      {config.pan_test_dir}")


if __name__ == "__main__":
    main()
