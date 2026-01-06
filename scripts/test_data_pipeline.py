"""Data pipeline testing module for UniMER dataset validation."""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.getcwd())

from configs.model_base import get_config
from unimernet.data.dataset import UniMERCollator, UniMERDataset, get_dataloader
from unimernet.data.transforms import (
    FormulaImageProcessor,
    get_train_transforms,
)

# Constants
IMAGE_MEAN = np.array([0.7931, 0.7931, 0.7931])
IMAGE_STD = np.array([0.1738, 0.1738, 0.1738])
PATCH_SIZE = 32
IGNORE_INDEX = -100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DataPipelineTester:
    """Comprehensive tester for the data pipeline components."""

    def __init__(self):
        self.config = get_config()
        self.tokenizer = self._setup_tokenizer()
        self.dataset = self._load_dataset()
        self.collator = UniMERCollator(pad_token_id=self.tokenizer.pad_token_id)

    def _setup_tokenizer(self) -> PreTrainedTokenizerFast:
        """Initialize and configure the tokenizer with special tokens."""
        tokenizer_file = os.path.join(
            self.config.model.tokenizer_path, "tokenizer.json"
        )

        if os.path.exists(tokenizer_file):
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        else:
            logger.warning(
                f"Tokenizer not found at {tokenizer_file}. Using default."
            )
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

        special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
        }
        tokens_to_add = {
            k: v for k, v in special_tokens.items() if getattr(tokenizer, k) is None
        }

        if tokens_to_add:
            tokenizer.add_special_tokens(tokens_to_add)

        return tokenizer

    def _load_dataset(self) -> UniMERDataset:
        """Load the training dataset."""
        dataset_path = self.config.train_data.dataset_path

        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path {dataset_path} does not exist.")
            sys.exit(1)

        return UniMERDataset(
            dataset_path=dataset_path,
            tokenizer=self.tokenizer,
            stage="train",
            image_size=self.config.train_data.image_size,
            max_seq_len=self.config.train_data.max_seq_len,
        )

    def test_config_consistency(self) -> None:
        """Verify config matches expected paper parameters."""
        logger.info("TEST: Config Consistency")

        assert self.config.model.max_seq_len == 1536, "Max sequence length should be 1536"

        h, w = self.config.train_data.image_size
        if h % PATCH_SIZE != 0 or w % PATCH_SIZE != 0:
            logger.warning(
                f"Image size {h}x{w} is not divisible by {PATCH_SIZE} (Swin Patch Stride)."
            )
        else:
            logger.info(f"Image size {h}x{w} is valid (divisible by {PATCH_SIZE}).")

    def test_tokenizer_special_tokens(self) -> None:
        """Verify special tokens are correctly assigned."""
        logger.info("TEST: Tokenizer Special Tokens")

        assert self.tokenizer.pad_token_id is not None, "PAD token missing"
        assert self.tokenizer.bos_token_id is not None, "BOS token missing"
        assert self.tokenizer.eos_token_id is not None, "EOS token missing"

        logger.info(
            f"PAD: {self.tokenizer.pad_token_id}, "
            f"BOS: {self.tokenizer.bos_token_id}, "
            f"EOS: {self.tokenizer.eos_token_id}"
        )

    def test_dataset_item_integrity(self) -> None:
        """Check a single dataset item for correct keys and content."""
        logger.info("TEST: Dataset Item Integrity")

        item = self.dataset[0]
        required_keys = ["pixel_values", "input_ids", "text_str"]

        for key in required_keys:
            assert key in item, f"Missing key {key} in dataset item"

        input_ids = item["input_ids"]
        assert input_ids[0] == self.tokenizer.bos_token_id, "Input IDs must start with BOS"
        assert input_ids[-1] == self.tokenizer.eos_token_id, "Input IDs must end with EOS"

        logger.info("Dataset item keys and special tokens validated.")

    def test_collator_logic(self) -> None:
        """Verify batch padding and label masking behavior."""
        logger.info("TEST: Collator Logic (Padding & Masking)")

        batch = [self.dataset[0], self.dataset[1]]
        collated = self.collator(batch)
        labels = collated["labels"]

        input_ids_list = [item["input_ids"] for item in batch]
        padded_manual = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        assert labels.shape == padded_manual.shape, (
            f"Label shape {labels.shape} mismatch with manual pad {padded_manual.shape}"
        )

        is_pad = padded_manual == self.tokenizer.pad_token_id

        if is_pad.any():
            assert torch.all(labels[is_pad] == IGNORE_INDEX), (
                "Labels must be -100 where input is PAD"
            )

        assert torch.all(labels[~is_pad] == padded_manual[~is_pad]), (
            "Labels must match input_ids at non-pad positions"
        )

        logger.info("Collator correctly pads inputs and masks labels.")

    def test_image_augmentations_visual(self, n_samples: int = 15) -> None:
        """Generate a visual debug image comparing clean and augmented inputs."""
        logger.info("TEST: Augmentation Visual Inspection")

        eval_processor = FormulaImageProcessor(
            image_size=self.config.train_data.image_size
        )
        train_transform = get_train_transforms(
            image_size=self.config.train_data.image_size
        )

        n_samples = min(n_samples, len(self.dataset))
        fig, axs = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))

        for i in range(n_samples):
            raw_item = self.dataset.dataset[i]
            raw_img = raw_item["image"]
            text_str = raw_item["text"]

            eval_img_np = eval_processor.prepare_input(raw_img, random_padding=False)

            aug_img_tensor = train_transform(raw_img)
            aug_img_np = aug_img_tensor.permute(1, 2, 0).numpy()
            aug_img_denorm = np.clip(aug_img_np * IMAGE_STD + IMAGE_MEAN, 0, 1)

            axs[i, 0].imshow(eval_img_np)
            axs[i, 0].set_title(f"Sample {i}: Clean Input")
            axs[i, 0].axis("off")
            axs[i, 0].text(0, -20, f"Label: {text_str[:40]}...", fontsize=8, wrap=True)

            axs[i, 1].imshow(aug_img_denorm, cmap="gray")
            axs[i, 1].set_title(f"Sample {i}: Augmented (Train)")
            axs[i, 1].axis("off")

        plt.tight_layout()
        save_path = "data_visual_debug.png"
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved visual debug to {save_path}")

    def test_dataloader_iteration(self, num_batches: int = 3) -> None:
        """Iterate through batches to catch runtime errors."""
        logger.info("TEST: DataLoader Iteration")

        dataloader = get_dataloader(self.config, stage="train", shuffle=False)

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            p_min = batch["pixel_values"].min().item()
            p_max = batch["pixel_values"].max().item()

            assert batch["pixel_values"].dtype == torch.float32
            assert batch["labels"].dtype == torch.long

            logger.info(
                f"Batch {i} - Shape: {batch['pixel_values'].shape}, "
                f"Range: [{p_min:.2f}, {p_max:.2f}]"
            )

        logger.info("DataLoader iteration successful.")

    def run_all(self) -> None:
        """Execute all pipeline tests."""
        logger.info("=== Starting Comprehensive Data Pipeline Test ===")

        try:
            self.test_config_consistency()
            self.test_tokenizer_special_tokens()
            self.test_dataset_item_integrity()
            self.test_collator_logic()
            self.test_dataloader_iteration()
            self.test_image_augmentations_visual()
            logger.info("=== All Tests Passed Successfully ===")

        except AssertionError as e:
            logger.error(f"Test Failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    tester = DataPipelineTester()
    tester.run_all()