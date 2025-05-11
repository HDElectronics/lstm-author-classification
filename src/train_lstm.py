
import logging
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.load import load_embeddings
from src.lstm_model import LSTMClassifier
import numpy as np
import wandb

def train_lstm_main():
    """
    Train an LSTM classifier on precomputed embeddings.
    Logs metrics to Weights & Biases and prints training progress.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LSTM training")

    # Load embeddings, labels, and author mapping
    emb_path = "dataset/embeddings/dataset.npz"
    logger.info(f"Loading dataset from {emb_path}")
     # Load embeddings, labels, and (optional) author names
    data    = np.load(emb_path)
    X       = data["embeddings"]   # (N, D)
    y_orig  = data["labels"]       # (N,)
    authors = data.get("authors", None)

    # Remap labels into a contiguous 0…(C−1) range and set num_classes
    unique_labels = np.unique(y_orig)
    label2idx     = {lbl: i for i, lbl in enumerate(unique_labels)}
    y             = np.array([label2idx[l] for l in y_orig], dtype=np.int64)
    num_classes   = len(unique_labels)

    logger.info(f"Found {num_classes} classes: {unique_labels}")

    # Hyperparameters
    config = {
        "hidden_dim": 256,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 150
    }
    # Initialize W&B
    wandb.init(project="lstm-author-classification", config=config)
    wandb.run.name = f"lstm_e{config['epochs']}_bs{config['batch_size']}_lr{config['learning_rate']}"
    wandb.watch(LSTMClassifier(X.shape[1], config["hidden_dim"], num_classes), log="all")

    # Prepare training tensors
    logger.info("Preparing training tensors")
    tensor_x = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,D)
    tensor_y = torch.tensor(y, dtype=torch.long)

    # Initialize model, optimizer, and loss
    logger.info("Initializing model and optimizer")
    model = LSTMClassifier(X.shape[1], config["hidden_dim"], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop (no batching)
    logger.info("Starting training loop")
    for epoch in range(1, config["epochs"] + 1):
        # Forward + backward on full dataset
        preds = model(tensor_x)
        loss = criterion(preds, tensor_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = loss.item()
        logger.info(f"Epoch {epoch}/{config['epochs']} — Avg Loss: {avg_loss:.4f}")
        # Log to W&B
        wandb.log({"epoch": epoch, "avg_loss": avg_loss})
    # Save trained model checkpoint with author mapping
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "lstm_classifier.pt")
    checkpoint = {"model_state_dict": model.state_dict(), "authors": authors}
    torch.save(checkpoint, model_path)
    logger.info(f"Saved model checkpoint to {model_path}")
    # Log model artifact to W&B
    wandb.save(model_path)
    wandb.finish()
    logger.info("Training complete")
