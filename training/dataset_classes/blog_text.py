import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer


class BlogAuthorshipDataset(Dataset):
    """Identification-format dataset for Blog Authorship Corpus"""
    
    def __init__(
        self,
        csv_path: str,
        author_to_idx: Optional[Dict[str, int]] = None,
        min_docs_per_author: int = 5,
        allowed_authors: Optional[List[str]] = None,
        split_type: str = "train",
        docs_by_author: Optional[Dict[str, List[Dict]]] = None,
    ):
        """
        Args:
            csv_path: Path to blogtext.csv
            author_to_idx: Precomputed author → idx mapping (for val/test consistency)
            min_docs_per_author: Filter authors with fewer docs
            allowed_authors: Whitelist of authors to include (enforces OSR integrity)
            split_type: "train", "val", "test", or "full"
            docs_by_author: Pre-parsed mapping (avoids re-parsing for val/test)
        """
        self.csv_path = Path(csv_path)
        self.split_type = split_type
        
        # Parse or reuse author→docs mapping
        if docs_by_author is not None:
            self.docs_by_author = docs_by_author
        else:
            self.docs_by_author = self._parse_csv(csv_path)
        
        # Filter sparse authors FIRST (before any mapping)
        if min_docs_per_author > 0:
            self.docs_by_author = {
                a: docs for a, docs in self.docs_by_author.items()
                if len(docs) >= min_docs_per_author
            }
        
        # Apply author whitelist (enforces OSR integrity)
        if allowed_authors is not None:
            self.docs_by_author = {
                a: docs for a, docs in self.docs_by_author.items()
                if a in allowed_authors
            }
        
        # Build flat lists with CONSISTENT ORDERING
        self.texts: List[str] = []
        self.authors: List[str] = []
        self.metadata: List[Dict] = []  # gender, age, topic, etc.
        
        for author, docs in self.docs_by_author.items():
            for doc in docs:
                self.texts.append(doc["text"])
                self.authors.append(author)
                self.metadata.append({
                    "gender": doc.get("gender", "unknown"),
                    "age": doc.get("age", -1),
                    "topic": doc.get("topic", "unknown"),
                    "sign": doc.get("sign", "unknown"),
                    "date": doc.get("date", "unknown"),
                })
        
        # Build label map ONLY from this split's authors
        if author_to_idx is not None:
            self.author_to_idx = author_to_idx
        else:
            unique_authors = sorted(set(self.authors))  # Deterministic ordering
            self.author_to_idx = {a: i for i, a in enumerate(unique_authors)}
        
        # Precompute labels (NO runtime lookup → prevents -1 leakage in train)
        self.labels: List[int] = [
            self.author_to_idx[author]  # Guaranteed ∈ [0, N-1] for train set
            for author in self.authors
        ]
        
        self.num_classes = len(self.author_to_idx)
        print(f"[{split_type}] {len(self.texts)} docs | {self.num_classes} authors | "
              f"min_docs={min_docs_per_author}")

    def _parse_csv(self, csv_path: str) -> Dict[str, List[Dict]]:
        """Parse blogtext.csv → {author_id: [doc1, doc2, ...]}"""
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        
        docs_by_author = defaultdict(list)
        for _, row in df.iterrows():
            try:
                author_id = str(int(row["id"]))  # Ensure string ID for consistency
                doc = {
                    "text": str(row["text"]).strip(),
                    "gender": str(row["gender"]).strip(),
                    "age": int(row["age"]),
                    "topic": str(row["topic"]).strip(),
                    "sign": str(row["sign"]).strip(),
                    "date": str(row["date"]).strip(),
                }
                if doc["text"]:  # Skip empty texts
                    docs_by_author[author_id].append(doc)
            except Exception as e:
                continue  # Skip malformed rows
        
        print(f"✓ Parsed {len(df)} rows → {len(docs_by_author)} authors")
        return docs_by_author

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "text": self.texts[index],
            "label": self.labels[index],          # Guaranteed valid for train set
            "author_id": self.authors[index],     # For debugging/gallery construction
            "gender": self.metadata[index]["gender"],
            "age": self.metadata[index]["age"],
            "topic": self.metadata[index]["topic"],
        }


class BlogAuthorshipDataModule(pl.LightningDataModule):
    """Author-disjoint splits for strict OSR training on Blog Authorship Corpus"""
    
    def __init__(
        self,
        csv_path: str = "/app/datasets/blog_authorship_corpus/blogtext.csv",
        test_csv_path: Optional[str] = None,  # Separate test file (optional)
        batch_size: int = 64,
        num_workers: int = 16,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        min_docs_per_author: int = 10,
        train_authors: int = 1000,
        val_authors: int = 1000,        # Total val authors (gallery + probes)
        val_probe_authors: int = 1000,  # Will be split 50/50 in/out gallery
        test_authors: int = 1000,
        predict_on_split: str = "val",
        seed: int = 42,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.test_csv_path = test_csv_path  # May be None → use csv_path with disjointness check
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.min_docs_per_author = min_docs_per_author
        self.train_authors = train_authors
        self.val_authors = val_authors
        self.val_probe_authors = val_probe_authors
        self.test_authors = test_authors
        self.predict_on_split = predict_on_split
        self.seed = seed
        
        self.tokenizer = None
        self.author_to_idx: Optional[Dict[str, int]] = None
        self.train_author_list: List[str] = []
        self.val_gallery_authors_list: List[str] = []
        self.val_probe_authors_list: List[str] = []
        self.test_author_list: List[str] = []

    def setup(self, stage: Optional[str] = None):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Step 1: Build full author → docs mapping from CSV
        full_dataset = BlogAuthorshipDataset(
            self.csv_path,
            min_docs_per_author=self.min_docs_per_author,
            split_type="full"
        )
        docs_by_author = full_dataset.docs_by_author
        all_authors = list(full_dataset.author_to_idx.keys())
        random.seed(42)
        random.shuffle(all_authors)
        
        # Step 2: Author-level splitting (CRITICAL for OSR integrity)
        self.train_author_list = all_authors[:self.train_authors]
        self.val_authors_list = all_authors[
            self.train_authors : self.train_authors + self.val_authors
        ]
        # Val probes: 50% in-gallery, 50% out-of-gallery
        self.val_in_gallery = self.val_authors_list[:self.val_probe_authors // 2]
        self.val_out_gallery = self.val_authors_list[self.val_probe_authors // 2:]
        
        # Step 3: Build label map from TRAIN authors only
        self.author_to_idx = {a: i for i, a in enumerate(self.train_author_list)}
        
        # Step 4: Create datasets
        self.train_dataset = BlogAuthorshipDataset(
            self.csv_path,
            author_to_idx=self.author_to_idx,
            min_docs_per_author=self.min_docs_per_author,
            allowed_authors=self.train_author_list,
            split_type="train",
            docs_by_author=docs_by_author,
        )
        
        # self.val_dataset = BlogAuthorshipDataset(
        #     self.csv_path,
        #     author_to_idx=self.author_to_idx,  # Maps ONLY train authors (OOG → -1)
        #     min_docs_per_author=self.min_docs_per_author,
        #     allowed_authors=self.val_authors_list,
        #     split_type="val",
        #     docs_by_author=docs_by_author,
        # )
        
        # # Step 5: Test dataset - parse test CSV separately (strict author disjointness)
        # # If no separate test file provided, reuse main CSV but ensure author disjointness
        # test_csv_path = getattr(self, "test_csv_path", self.csv_path)
        
        # full_test_dataset = BlogAuthorshipDataset(
        #     test_csv_path,
        #     min_docs_per_author=self.min_docs_per_author,
        #     split_type="full"
        # )
        
        # # Filter test authors to ensure disjointness from train/val
        # test_authors_pool = [
        #     a for a in full_test_dataset.author_to_idx.keys()
        #     if a not in self.train_author_list and a not in self.val_authors_list
        # ]
        
        # # If insufficient disjoint authors, fall back to shuffling full test pool
        # if len(test_authors_pool) < self.test_authors:
        #     print(f"⚠️  Insufficient disjoint test authors ({len(test_authors_pool)} < {self.test_authors}), "
        #         f"using shuffled full test pool")
        #     test_authors_pool = list(full_test_dataset.author_to_idx.keys())
        
        # random.seed(42)
        # random.shuffle(test_authors_pool)
        
        # self.test_authors_list = test_authors_pool[:self.test_authors]
        # self.test_in_gallery = self.test_authors_list[:self.test_probe_authors // 2]
        # self.test_out_gallery = self.test_authors_list[self.test_probe_authors // 2:]
        
        # self.test_dataset = BlogAuthorshipDataset(
        #     test_csv_path,
        #     author_to_idx=self.author_to_idx,  # All test authors should be unseen → label=-1
        #     min_docs_per_author=self.min_docs_per_author,
        #     allowed_authors=self.test_authors_list,
        #     split_type="test",
        #     docs_by_author=full_test_dataset.docs_by_author,
        # )
        
        # Step 6: Verify OSR integrity
        # self._verify_splits()
        
        # print(f"\n✓ Train authors: {len(self.train_author_list)} → {len(self.train_dataset)} docs")
        # print(f"✓ Val authors: {len(self.val_authors_list)} "
        #     f"({len(self.val_in_gallery)} in-gallery, {len(self.val_out_gallery)} OOG)")
        # print(f"✓ Test authors: {len(self.test_authors_list)} "
        #     f"({len(self.test_in_gallery)} in-gallery, {len(self.test_out_gallery)} OOG)")
        
    
    def _verify_splits(self):
        """Critical: Verify no author leakage between splits"""
        train_set = set(self.train_author_list)
        val_gallery_set = set(self.val_gallery_authors_list)
        val_probe_set = set(self.val_probe_authors_list)
        test_set = set(self.test_author_list)
        
        assert train_set.isdisjoint(val_gallery_set), "❌ Train/Val gallery leakage!"
        assert train_set.isdisjoint(val_probe_set), "❌ Train/Val probe leakage!"
        assert train_set.isdisjoint(test_set), "❌ Train/Test leakage!"
        assert val_gallery_set.isdisjoint(test_set), "❌ Val/Test gallery leakage!"
        assert len(val_probe_set & val_gallery_set) == len(self.val_probe_authors_list) // 2, \
            "❌ Val probe in-gallery count mismatch!"
        
        # Verify no -1 labels in train set
        assert -1 not in self.train_dataset.labels, "❌ Train set contains label=-1!"
        
        print("✅ OSR integrity verified: all splits author-disjoint")


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
        
        if "label" in batch[0]:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            return input_dict, labels
        
        return input_dict

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        if self.predict_on_split == 'test':
            ds = self.test_dataset 
        elif self.predict_on_split == 'val':
            ds = self.val_dataset
        elif self.predict_on_split == 'train':
            ds = self.train_dataset
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
