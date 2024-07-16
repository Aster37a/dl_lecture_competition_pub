import torch
import numpy as np

from src.datasets import ThingsMEGDataset

def inspect_dataset(data_dir):
    train_set = ThingsMEGDataset("train", data_dir)
    val_set = ThingsMEGDataset("val", data_dir)
    test_set = ThingsMEGDataset("test", data_dir)

    datasets = {"train": train_set, "val": val_set, "test": test_set}

    for split, dataset in datasets.items():
        print(f"{split.upper()} DATASET")
        print(f"Number of samples: {len(dataset)}")
        print(f"Shape of X: {dataset.X.shape}")
        if hasattr(dataset, 'y'):
            print(f"Shape of y: {dataset.y.shape}")
            unique, counts = np.unique(dataset.y.numpy(), return_counts=True)
            print(f"Class distribution: {dict(zip(unique, counts))}")
        print()

if __name__ == "__main__":
    data_dir = "data"  # データディレクトリを指定
    inspect_dataset(data_dir)
