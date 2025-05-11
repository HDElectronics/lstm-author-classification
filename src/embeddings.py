from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_and_save_embeddings(
    texts,
    model_name="aubmindlab/bert-large-arabertv02",
    output_path="dataset/embeddings.npz"
):
    """
    Generate AraBERT [CLS] embeddings for a list of texts and save to a .npz file.

    Args:
        texts (List[str]): Input texts to embed.
        model_name (str): HuggingFace AraBERT model identifier.
        output_path (str): Path to save the embeddings archive (.npz).
    """
    logger.info(f"Starting embedding of {len(texts)} texts using model {model_name}")
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    with torch.no_grad():
        for idx, text in enumerate(texts):
            preprocessed = arabert_prep.preprocess(text)
            inputs = tokenizer(preprocessed, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_embedding)
            if idx and idx % 50 == 0:
                logger.debug(f"Embedded {idx} texts")

    embeddings = np.stack(embeddings)
    embeddings = np.stack(embeddings)
    # ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, embeddings=embeddings)
    logger.info(f"Saved {embeddings.shape[0]} embeddings to {output_path} with shape {embeddings.shape}")
