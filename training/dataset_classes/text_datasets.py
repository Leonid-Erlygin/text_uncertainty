from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import json
import pytorch_lightning as pl
import torch
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoTokenizer
import numpy as np


class Clinc150Dataset(Dataset):
    def __init__(
        self,
        json_path: str,
        split: str = "train",
        label_map: Dict[str, int] = None,
        include_oos: bool = True,
    ):
        """
        Args:
            json_path (str): Path to data_full.json
            split (str): One of "train", "val", "test"
            label_map (dict): Precomputed mapping from intent name → class index.
                              If None, built from in-scope intents in this split.
            include_oos (bool): If False, filter out OOS samples (label == "oos")
        """
        self.json_path = Path(json_path)
        self.split = split
        self.include_oos = include_oos

        # Map split names to JSON keys
        key_map = {
            "train": "train",
            "val": "oos_val",
            "test": "oos_test",
        }
        split_key = key_map[split]

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_data = data[split_key]  # List of [text, label_str]

        texts: List[str] = []
        labels_str: List[str] = []

        for text, label in raw_data:
            if not include_oos and label == "oos":
                continue
            texts.append(text)
            labels_str.append(label)

        # Build or use label map (only from in-scope labels)
        if label_map is not None:
            self.label_map = label_map
        else:
            # Build from in-scope labels only
            in_scope_labels = [lbl for lbl in labels_str if lbl != "oos"]
            unique_labels = sorted(set(in_scope_labels))
            self.label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

        self.num_classes = len(self.label_map)

        # Store raw data
        self.texts = texts
        self.labels_str = labels_str

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index) -> Dict[str, Any]:
        label_str = self.labels_str[index]
        if label_str == "oos":
            label = -1
        else:
            label = self.label_map[label_str]
        return {
            "text": self.texts[index],
            "label": label,
        }

class Clinc150DataModule(pl.LightningDataModule):
    def __init__(
        self,
        json_path: str = "",
        batch_size: int = 128,
        num_workers: int = 4,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 64,
    ):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

        # Tokenizer will be initialized in setup (main process)
        self.tokenizer = None
        self.label_map: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Build label map from full training set (in-scope only)
        full_train = Clinc150Dataset(
            self.json_path, split="train", include_oos=False
        )
        self.label_map = full_train.label_map

        # Training: exclude OOS
        self.train_dataset = Clinc150Dataset(
            self.json_path,
            split="train",
            label_map=self.label_map,
            include_oos=False,
        )

        # Validation and test: include OOS (for evaluation)
        self.val_dataset = Clinc150Dataset(
            self.json_path,
            split="val",
            label_map=self.label_map,
            include_oos=True,
        )
        self.test_dataset = Clinc150Dataset(
            self.json_path,
            split="test",
            label_map=self.label_map,
            include_oos=True,
        )

    def collate_fn(self, batch: List[Dict[str, Any]]):
        texts = [item["text"] for item in batch]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_dict = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        # Include labels if present
        if "label" in batch[0]:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            return input_dict, labels
        else:
            return input_dict

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #         num_workers=self.num_workers,
    #         collate_fn=self.collate_fn,
    #     )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class TextDatasets(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        batch_size: int = 16,
        num_workers: int = 4,
        tokenizer_name: str = "bert-base-uncased",
        predict_dataset: Dataset = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.train_dataset = train_dataset
        self.predict_dataset = predict_dataset
        # Instantiate tokenizer once (in main process)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage=None):
        # Build dataset using config
        pass

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]

        tokenized = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )

        tokenized_inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        # Check if labels are present (i.e., training/val mode)
        if "label" in batch[0]:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            return tokenized_inputs, labels
        else:
            # Prediction mode: no labels
            return tokenized_inputs

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # <-- critical
        )


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        tokenizer_name: str = "bert-base-uncased",  # will be used only for info, not tokenization here
    ):
        # Load data
        df = pd.read_csv(
            Path(root_dir) / "in_distribution_train.csv",
            header=None,
            usecols=[0, 2],
            names=["class", "text"],
            quotechar='"',
        )
        unique_classes = sorted(np.unique(df["class"].values))  # [2, 3, 5, 6, 7, 8]
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        df["class"] = df["class"].map(class_to_idx)
        df["text"] = df["text"].str.strip("\"'")

        self.texts = df["text"].values
        self.labels = df["class"].values
        self.tokenizer_name = (
            tokenizer_name  # store for compatibility (e.g., num classes, etc.)
        )

    def __getitem__(self, index):
        # Return RAW text and label — NO tokenization here
        return {"text": self.texts[index], "label": self.labels[index]}

    def __len__(self):
        return len(self.labels)


class TextPredictionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        tokenizer_name: str = "bert-base-uncased",  # will be used only for info, not tokenization here
    ):
        root_dir = Path(root_dir)
        paths = []
        with open(root_dir / "meta" / f"{root_dir.parts[-1]}_face_tid_mid.txt") as fd:
            for line in fd:
                paths.append(line.split(" ")[0])
        self.texts = []
        for path in paths:
            with open(root_dir / path) as fd:
                self.texts.append(fd.read())

    def __getitem__(self, index):
        # Return RAW text and label — NO tokenization here
        return {"text": self.texts[index]}

    def __len__(self):
        return len(self.texts)
