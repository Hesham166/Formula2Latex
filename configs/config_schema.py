from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    arch: str = "unimernet"
    model_name: str = ""
    tokenizer_path: str = "./data/"
    max_seq_len: int = 1536

    encoder_layers: int = 6
    encoder_heads: int = 8

    decoder_layers: int = 8
    decoder_heads: int = 8

    hidden_dim: int = 1024
    vocab_size: int = 0


@dataclass
class DataConfig:
    dataset_path: str
    split: str = "train"
    image_size: List[int] = field(default_factory=lambda: [192, 672])
    max_seq_len: int = 1536
    sample_rate: float = 1.0


@dataclass
class TrainingConfig:
    output_dir: str
    
    # Optimization
    lr_sched: str = "cosine"
    init_lr: float = 1e-4
    min_lr: float = 1e-8
    warmup_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_steps: int = 5000
   
    # Loop controls
    batch_size_train: int = 8
    batch_size_eval: int = 8
    max_iters: int = 300000
    eval_interval: int = 5000
    log_interval: int = 100
   
    # Hardware / Distributed
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 8
    amp: bool = True # Automatic Mixed Precision
   
    # Generation (for validation)
    generate_temperature: float = 0.0

@dataclass
class ExperimentConfig:
    model: ModelConfig
    train_data: DataConfig
    val_data: DataConfig
    training: TrainingConfig
