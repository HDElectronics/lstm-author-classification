import numpy as np
import logging

def load_embeddings(npz_path="dataset/embeddings.npz"):
    """
    Load embeddings array from a .npz file.

    Args:
        npz_path (str): Path to the .npz file containing 'embeddings'.
    Returns:
        np.ndarray: Loaded embeddings of shape (N, embedding_dim).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading embeddings from {npz_path}")
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    logger.info(f"Loaded embeddings with shape {embeddings.shape}")
    return embeddings
