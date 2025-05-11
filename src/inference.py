"""
Inference module: load embeddings, load trained LSTM model, and run predictions.
"""
import logging
import os
import torch
import numpy as np
from src.load import load_embeddings
from src.lstm_model import LSTMClassifier

logger = logging.getLogger(__name__)

def run_inference(embeddings_path="dataset/embeddings.npz", model_path=None):
    """
    Run inference on precomputed embeddings using a trained LSTM model.

    Args:
        embeddings_path (str): NPZ path containing 'embeddings'.
        model_path (str | None): Path to .pt model state dict. If None, uses randomly initialized model.
    Returns:
        np.ndarray: Predicted class indices for each sample.
    """
    logger.info(f"Running inference with embeddings from {embeddings_path}")
    embeddings = load_embeddings(embeddings_path)
    embedding_dim = embeddings.shape[1]
    hidden_dim = 256  # Should match training
    # Prepare data tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1)  # shape (N,1,D)

    # Load model checkpoint and author mapping
    if model_path is None:
        model_path = os.path.join("models", "lstm_classifier.pt")
    logger.info(f"Loading model checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    authors = checkpoint.get("authors", None)
    num_classes = len(authors) if authors is not None else checkpoint.get("num_classes", None)

    # Initialize and load model
    logger.info(f"Initializing LSTMClassifier(emb_dim={embedding_dim}, hid_dim={hidden_dim}, classes={num_classes})")
    model = LSTMClassifier(embedding_dim, hidden_dim, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    with torch.no_grad():
        outputs = model(embeddings_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    # Map indices to author names if available
    if authors is not None:
        pred_names = np.array(authors)[predictions]
        logger.info(f"Inference completed. Predicted authors: {pred_names}")
        return pred_names
    else:
        logger.info(f"Inference completed. Predictions: {predictions}")
        return predictions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
