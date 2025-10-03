import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    X, y = fetch_20newsgroups(return_X_y=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(X, convert_to_numpy=True, show_progress_bar=True)
    np.save("docs/data/newsgroups/generated/X.npy", X)
    np.save("docs/data/newsgroups/generated/y.npy", y)
