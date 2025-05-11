
import sys
import logging
import re
from src.embeddings import generate_and_save_embeddings
from src.train_lstm import train_lstm_main
from src.inference import run_inference

def main():
    """
    Entry point: choose 'embeddings', 'train', or 'inference'.
    """
    logger = logging.getLogger(__name__)
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py [embeddings|train|inference] [optional: text_file.txt]")
        return
    command = sys.argv[1]
    logger.info(f"Command selected: {command}")
    if command == "embeddings":
        logger.info("Building embeddings dataset from folder 'dataset/raw'")
        from src.data_loader import build_dataset
        build_dataset(  raw_dir="dataset/raw",
                        output_path="dataset/embeddings/dataset.npz")
        logger.info("Embeddings dataset built successfully")
    elif command == "train":
        logger.info("Starting LSTM training")
        train_lstm_main()
        logger.info("LSTM training completed")
    elif command == "inference":
        if len(sys.argv) < 3:
            logger.error("Usage: python main.py inference <text_file.txt>")
            return
        text_file = sys.argv[2]
        logger.info(f"Loading texts from file: {text_file}")
        with open(text_file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(texts)} texts for inference")
        # Generate embeddings for the input texts
        from src.embeddings import generate_and_save_embeddings
        infer_path = "dataset/infer_embeddings.npz"
        generate_and_save_embeddings(texts, output_path=infer_path)
        logger.info("Embeddings generated for inference")
        run_inference(embeddings_path=infer_path)
    else:
        logger.error(f"Unknown command: {command}")

if __name__ == "__main__":
    # Colored logging formatter
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"

    class CustomFormatter(logging.Formatter):
        """Logging Formatter to add colors based on log level"""
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        LEVEL_COLORS = {
            logging.INFO: GREEN,
            logging.WARNING: YELLOW,
            logging.ERROR: RED,
        }

        def format(self, record):
            log_fmt = self.FORMAT
            message = logging.Formatter(log_fmt).format(record)
            color = self.LEVEL_COLORS.get(record.levelno, RESET)
            return f"{color}{message}{RESET}"

    # Configure root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    main()
