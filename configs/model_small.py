from configs.config_schema import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig


def get_config() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            model_name="./models/unimernet_small",
            tokenizer_path="./data",
            max_seq_len=1536,
            encoder_layers=6,
            decoder_layers=8,
            hidden_dim=768
        ),
        train_data=DataConfig(
            dataset_path="./data",
            split="train",
            image_size=[192, 672],
            max_seq_len=1536
        ),
        val_data=DataConfig(
            dataset_path="./data",
            split="test",
            image_size=[192, 672],
            max_seq_len=1536
        ),
        training=TrainingConfig(
            output_dir="./outputs_unimernet/unimernet_small",
            batch_size_train=64,
            batch_size_eval=64,
            max_iters=300000,
        )
    )
