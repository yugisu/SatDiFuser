import numpy as np
import torch
import faiss


class FAISSRetriever:
    """Uses cosine similarity; assumes database and query embeddings are L2-normalized."""

    def __init__(self, database_embeddings: torch.Tensor):
        self.database_embeddings = database_embeddings

        self.index = faiss.IndexFlatIP(database_embeddings.shape[1])

        g_np = np.ascontiguousarray(
            database_embeddings.detach().cpu().numpy().astype(np.float32)
        )
        self.index.add(g_np)

    def search(self, query_embeddings: torch.Tensor, k: int = 10):
        q_np = np.ascontiguousarray(
            query_embeddings.detach().cpu().numpy().astype(np.float32)
        )
        distances, indices = self.index.search(q_np, k)
        return distances, indices
