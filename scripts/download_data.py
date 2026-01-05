from datasets import load_dataset


def download_unimer_dataset(target_dir: str = "./data"):
    dataset = load_dataset("deepcopy/UniMER")
    dataset.save_to_disk(target_dir)

    print(f"Downloaded UniMER dataset to {target_dir}")
    print(f"Splits: {dataset.keys()}")
    print(f"Number of train examples: {dataset['train'].num_rows}")
    print(f"Number of test examples: {dataset['test'].num_rows}")


if __name__ == "__main__":
    download_unimer_dataset()