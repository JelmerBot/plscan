from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("uoft-cs/cifar10")
    df = ds.data["train"].to_pandas()
    df.to_parquet("docs/data/cifar_10/sources/cifar10-train.parquet")
