import logging
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.load import load_embeddings
from src.lstm_model import LSTMClassifier
import numpy as np
import wandb
import random


def train_lstm_main():
    """
    Train an LSTM classifier on precomputed embeddings.
    Logs metrics to Weights & Biases and prints training progress.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LSTM training")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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
        "hidden_dim": 128,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 50,
        "dropout": 0.3,
        "weight_decay": 1e-5
    }
    # Initialize W&B
    wandb.init(project="lstm-author-classification", config=config)
    wand_run_name = f"lstm_e{config['epochs']}_bs{config['batch_size']}_lr{config['learning_rate']}"
    wandb.run.name = wand_run_name
    wandb.watch(LSTMClassifier( X.shape[1],
                                config["hidden_dim"],
                                dropout=config["dropout"],
                                num_classes=num_classes), log="all")

    # Split data into train, validation, and test sets (60/20/20)
    logger.info("Splitting data into train/val/test sets (60/20/20)")
    N = X.shape[0]
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_end = int(0.6 * N)
    val_end = int(0.8 * N)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    logger.info(f"Data split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Prepare DataLoaders
    tensor_x_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)
    tensor_x_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    tensor_y_val = torch.tensor(y_val, dtype=torch.long)
    tensor_x_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    tensor_y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model, optimizer, and loss function
    logger.info("Initializing model, optimizer, and loss function")
    model = LSTMClassifier(
        embedding_dim=X.shape[1],
        hidden_dim=128,
        num_classes=num_classes,
        dropout=0.3,
        num_layers=2,
        bidirectional=True,
        use_attention=True
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop with validation
    logger.info("Starting training loop")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                _, preds = torch.max(logits, dim=1)
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)
        val_loss /= total_val
        val_accuracy = correct_val / total_val

        logger.info(f"Epoch {epoch}/{config['epochs']} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            test_loss += loss.item() * xb.size(0)
            _, preds = torch.max(logits, dim=1)
            correct_test += (preds == yb).sum().item()
            total_test += yb.size(0)
    test_loss /= total_test
    test_accuracy = correct_test / total_test
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
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
