import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from typing import List

from unimernet.data.transforms import get_train_transforms, get_eval_transforms


class UniMERDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerFast,
        stage: str = "train",
        image_size: List[int] = [192, 672],
        max_seq_len: int = 1536,
    ):
        self.dataset = load_from_disk(dataset_path)[stage]
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_seq_len = max_seq_len
        
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

        if stage == "train":
            self.transform = get_train_transforms(image_size)
        elif stage == "test":
            self.transform = get_eval_transforms(image_size)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        pixel_values = self.transform(item['image'])
        text = item['text']

        # Encode with space reserved for BOS/EOS
        input_ids = self.tokenizer.encode(
            text,
            max_length=self.max_seq_len - 2,  # Reserve space
            truncation=True,
            add_special_tokens=False
        )

        needs_bos = not input_ids or input_ids[0] != self.bos_id
        needs_eos = not input_ids or input_ids[-1] != self.eos_id
        
        if needs_bos and needs_eos:
            input_ids = [self.bos_id] + input_ids + [self.eos_id]
        elif needs_bos:
            input_ids = [self.bos_id] + input_ids
        elif needs_eos:
            input_ids.append(self.eos_id)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "text_str": text,
        }


class UniMERCollator:
    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        input_ids = [item["input_ids"] for item in batch]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.pad_token_id
        )
        
        labels = padded_input_ids.clone()
        labels[labels == self.pad_token_id] = self.ignore_index

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "texts": [item["text_str"] for item in batch]
        }


def get_dataloader(
    config,
    stage: str = 'train',
    shuffle: bool = True
):
    tokenizer_file = os.path.join(config.model.tokenizer_path, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<pad>"
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = "<s>"
        
    data_config = config.train_data if stage == 'train' else config.val_data
    
    dataset = UniMERDataset(
        dataset_path=data_config.dataset_path,
        tokenizer=tokenizer,
        stage=stage,
        image_size=data_config.image_size,
        max_seq_len=data_config.max_seq_len
    )
    
    collator = UniMERCollator(pad_token_id=tokenizer.pad_token_id)
    
    batch_size = config.training.batch_size_train if stage == 'train' else config.training.batch_size_eval
    num_workers = config.training.num_workers
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )