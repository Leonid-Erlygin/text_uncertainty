from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
import numpy as np


class TextDatasets(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        batch_size: int = 16,
        num_workers: int = 4,
        tokenizer_name: str = "bert-base-uncased",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.train_dataset = train_dataset

        # Instantiate tokenizer once (in main process)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage=None):
        # Build dataset using config
        pass

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        tokenized = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )

        # Pack tokenized inputs into a dict (this plays the role of "images")
        tokenized_inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            # add "token_type_ids" if your tokenizer/bert uses them
        }

        # Return as (inputs_dict, labels) → matches (images, labels) unpacking
        return tokenized_inputs, labels

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
