import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer


class PANAuthorshipDataset(Dataset):
    """Identification-format dataset using TRUE author IDs from PAN splits"""
    
    def __init__(
        self,
        jsonl_path: str,
        author_to_idx: Optional[Dict[str, int]] = None,
        min_docs_per_author: int = 5,
        split_authors: Optional[List[str]] = None,
        split_type: str = "train",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.split_type = split_type
        
        # Convert verification pairs → true identification format
        docs_by_author = self._convert_to_identification_format(jsonl_path)
        
        # Filter sparse authors (critical for stable ArcFace training)
        if min_docs_per_author > 0:
            docs_by_author = {
                a: docs for a, docs in docs_by_author.items() 
                if len(docs) >= min_docs_per_author
            }
        
        # Apply author-level split constraint (enforces OSR integrity)
        if split_authors is not None:
            docs_by_author = {
                a: docs for a, docs in docs_by_author.items() 
                if a in split_authors
            }
        
        # Build flat lists
        self.texts: List[str] = []
        self.authors: List[str] = []
        for author, docs in docs_by_author.items():
            for doc in docs:
                self.texts.append(doc["text"])
                self.authors.append(author)
        
        # Build/reuse label map (author → idx)
        if author_to_idx is not None:
            self.author_to_idx = author_to_idx
        else:
            unique_authors = sorted(set(self.authors))
            self.author_to_idx = {a: i for i, a in enumerate(unique_authors)}
        
        self.num_classes = len(self.author_to_idx)
        print(f"[{split_type}] Loaded {len(self.texts)} docs from {self.num_classes} authors "
              f"(min_docs={min_docs_per_author})")

    def _convert_to_identification_format(
        self, 
        jsonl_path: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Build true author→docs mapping using explicit 'authors' field"""
        docs_by_author = defaultdict(list)
        total_pairs = 0
        sa_pairs = 0
        da_pairs = 0
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                pair = json.loads(line.strip())
                total_pairs += 1
                
                # Extract fields
                texts = pair["pair"]          # [text1, text2]
                authors = pair["authors"]     # [author1, author2]
                fandoms = pair.get("fandoms", ["unknown", "unknown"])
                is_same = pair["same"]        # boolean
                
                if is_same:
                    sa_pairs += 1
                    # Both texts belong to SAME author
                    author_id = authors[0]  # authors[0] == authors[1]
                    docs_by_author[author_id].append({
                        "text": texts[0],
                        "fandom": fandoms[0],
                        "pair_id": pair["id"]
                    })
                    docs_by_author[author_id].append({
                        "text": texts[1],
                        "fandom": fandoms[1],
                        "pair_id": pair["id"]
                    })
                else:
                    da_pairs += 1
                    # Different authors — add each text to its respective author
                    docs_by_author[authors[0]].append({
                        "text": texts[0],
                        "fandom": fandoms[0],
                        "pair_id": pair["id"]
                    })
                    docs_by_author[authors[1]].append({
                        "text": texts[1],
                        "fandom": fandoms[1],
                        "pair_id": pair["id"]
                    })
        
        print(f"✓ Parsed {total_pairs} pairs: {sa_pairs} SA, {da_pairs} DA")
        print(f"✓ Total unique authors: {len(docs_by_author)}")
        return docs_by_author

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        author = self.authors[index]
        label = self.author_to_idx.get(author, -1)  # -1 for unseen authors
        
        return {
            "text": self.texts[index],
            "label": label,
            "author_id": author,
        }


class PANDataModule(pl.LightningDataModule):
    """Author-disjoint splits for strict OSR training with TRUE author IDs"""
    
    def __init__(
        self,
        train_jsonl: str = "/app/datasets/pan/unseen_authors/xl/pan20-av-large-notest.jsonl",
        test_jsonl: str = "/app/datasets/pan/unseen_authors/xl/pan20-av-large-test.jsonl",
        batch_size: int = 64,
        num_workers: int = 16,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        min_docs_per_author: int = 5,
        train_authors: int = 15000,
        val_gallery_authors: int = 2000,
        val_probe_authors: int = 2000,
    ):
        super().__init__()
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.min_docs_per_author = min_docs_per_author
        self.train_authors = train_authors
        self.val_gallery_authors = val_gallery_authors
        self.val_probe_authors = val_probe_authors
        
        self.tokenizer = None
        self.author_to_idx: Optional[Dict[str, int]] = None
        self.train_author_list: List[str] = []
        self.val_gallery_authors_list: List[str] = []
        self.val_probe_authors_list: List[str] = []

    def setup(self, stage: Optional[str] = None):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Step 1: Build full author → docs mapping from train JSONL
        full_dataset = PANAuthorshipDataset(
            self.train_jsonl,
            min_docs_per_author=self.min_docs_per_author,
            split_type="full"
        )
        all_authors = list(full_dataset.author_to_idx.keys())
        random.seed(42)
        random.shuffle(all_authors)
        
        # Step 2: Author-level splitting (CRITICAL for OSR integrity)
        self.train_author_list = all_authors[:self.train_authors]
        self.val_gallery_authors_list = all_authors[
            self.train_authors : self.train_authors + self.val_gallery_authors
        ]
        # Val probes: 50% in-gallery, 50% out-of-gallery
        val_in_gallery = self.val_gallery_authors_list[:self.val_probe_authors // 2]
        val_out_gallery = all_authors[
            self.train_authors + self.val_gallery_authors : 
            self.train_authors + self.val_gallery_authors + self.val_probe_authors // 2
        ]
        self.val_probe_authors_list = val_in_gallery + val_out_gallery
        
        # Step 3: Build label map from TRAIN authors only
        self.author_to_idx = {a: i for i, a in enumerate(self.train_author_list)}
        
        # Step 4: Create datasets
        self.train_dataset = PANAuthorshipDataset(
            self.train_jsonl,
            author_to_idx=self.author_to_idx,
            min_docs_per_author=self.min_docs_per_author,
            split_authors=self.train_author_list,
            split_type="train",
        )
        
        self.val_dataset = PANAuthorshipDataset(
            self.train_jsonl,
            author_to_idx=self.author_to_idx,
            min_docs_per_author=self.min_docs_per_author,
            split_authors=self.val_probe_authors_list,
            split_type="val",
        )
        
        # Test dataset: use official unseen-authors split (all authors should be unseen → label=-1)
        self.test_dataset = PANAuthorshipDataset(
            self.test_jsonl,
            author_to_idx=self.author_to_idx,
            min_docs_per_author=self.min_docs_per_author,
            split_type="test",
        )
        
        print(f"\n✓ Train authors: {len(self.train_author_list)} → {len(self.train_dataset)} docs")
        print(f"✓ Val gallery authors: {len(self.val_gallery_authors_list)}")
        print(f"✓ Val probe authors: {len(self.val_probe_authors_list)} "
              f"({len(val_in_gallery)} in-gallery, {len(val_out_gallery)} out-of-gallery)")
        print(f"✓ Test authors (should be unseen): {self.test_dataset.num_classes} "
              f"→ {len(self.test_dataset)} docs (all mapped to label=-1)")

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

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()