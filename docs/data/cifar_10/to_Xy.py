import io
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image
from transformers import AutoProcessor, AutoModel

if __name__ == "__main__":
    df = pd.read_parquet("docs/data/cifar_10/sources/cifar10-train.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True
    )

    batch_size = 1024
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]
        images = [Image.open(io.BytesIO(img["bytes"])) for img in batch_df.img]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True)
            batch_embedding = model.get_image_features(
                **{k: v.to(device) for k, v in inputs.items()}
            )
        embeddings.append(batch_embedding.cpu())
    embedding = torch.cat(embeddings, dim=0)

    np.save(
        "docs/data/cifar_10/generated/X.npy",
        embedding.float().cpu().numpy().astype(np.float32),
    )
    np.save("docs/data/cifar_10/generated/y.npy", df.label.to_numpy())
