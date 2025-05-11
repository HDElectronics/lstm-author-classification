# LSTM Author Classification

This project implements an LSTM-based classifier for author identification using precomputed text embeddings. The code is designed for research and experimentation with deep learning models for text classification, leveraging PyTorch and Weights & Biases (W&B) for training and experiment tracking.

## Project Structure

```
lstm-project/
├── main.py                  # (Entry point, optional)
├── requirements.txt         # Python dependencies
├── dataset/
│   └── embeddings/
│       └── dataset.npz      # Precomputed embeddings and labels
├── models/                  # Saved model checkpoints
├── src/
│   ├── train_lstm.py        # Main training script
│   ├── lstm_model.py        # LSTMClassifier model definition
│   ├── load.py              # Data loading utilities
│   ├── ...                  # Other utilities
└── wandb/                   # W&B run logs and artifacts
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd lstm-project
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Weights & Biases (W&B):**
   - Sign up at [wandb.ai](https://wandb.ai/) and get your API key.
   - Run `wandb login` and paste your API key when prompted.

## Data Preparation

- Place your precomputed embeddings and labels in `dataset/embeddings/dataset.npz`.
- The `.npz` file should contain:
  - `embeddings`: NumPy array of shape `(N, D)`
  - `labels`: NumPy array of shape `(N,)`
  - (Optional) `authors`: Array of author names

## Training the Model

Run the main training script:

```bash
python -m src.train_lstm
```

- The script will:
  - Load the embeddings and labels
  - Split the data into train/validation/test sets (60/20/20)
  - Train an LSTM classifier with attention and class-weighted loss
  - Log metrics to W&B
  - Save the best model checkpoint to `models/best.pt` and a final checkpoint to `models/lstm_classifier.pt`

## Model Checkpoints

- `models/best.pt`: Best model (lowest validation loss)
- `models/lstm_classifier.pt`: Final model with author mapping

## Customization

- Hyperparameters (hidden size, batch size, learning rate, etc.) can be modified in `src/train_lstm.py` under the `config` dictionary.
- The model architecture can be changed in `src/lstm_model.py`.

## Inference

- Use the utilities in `src/inference.py` (if available) or load the saved model checkpoint for predictions.

## Logging & Experiment Tracking

- All training metrics and model artifacts are logged to your W&B project (`lstm-author-classification`).
- You can view your runs and compare experiments at [wandb.ai](https://wandb.ai/).

## Requirements

- Python 3.8+
- PyTorch
- numpy
- wandb

Install all dependencies with `pip install -r requirements.txt`.

## License

This project is for research and educational purposes.
