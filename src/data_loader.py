import os
import numpy as np
import torch
import logging
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModel

def build_dataset(raw_dir="dataset/raw", output_path="dataset/embeddings/dataset.npz",
                  model_name="aubmindlab/bert-large-arabertv02"):
    """
    Walk through `raw_dir` where each subfolder is an author, read all .txt files,
    generate AraBERT [CLS] embeddings, and save arrays to `output_path`:
      - embeddings: NumPy array of shape (N, hidden_dim)
      - labels:     NumPy array of shape (N,)
      - authors:    array of author folder names (string)
    Returns:
      embeddings, labels, author_names
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building dataset from raw_dir={raw_dir}")
    # collect texts and labels
    author_names = []
    texts = []
    labels = []
    # discover author folders only
    author_dirs = [d for d in sorted(os.listdir(raw_dir))
                   if os.path.isdir(os.path.join(raw_dir, d))]
    for label_idx, author in enumerate(author_dirs):
        author_names.append(author)
        author_path = os.path.join(raw_dir, author)
        # walk through nested folders to find .txt files
        for root, _, files in os.walk(author_path):
            for fname in files:
                if fname.lower().endswith('.txt'):
                    file_path = os.path.join(root, fname)
                    with open(file_path, encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        texts.append(text)
                        labels.append(label_idx)

    if not texts:
        logger.error(f"No text files found under {raw_dir}")
        raise ValueError(f"No text files found under {raw_dir}")

    # prepare model and tokenizer
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # generate embeddings
    embeddings = []
    logger.info(f"Generating embeddings for {len(texts)} texts using model {model_name}")
    with torch.no_grad():
        for idx, text in enumerate(texts):
            preprocessed = arabert_prep.preprocess(text)
            inputs = tokenizer(preprocessed, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            # CLS token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_emb)
            if idx and idx % 100 == 0:
                logger.debug(f"Processed {idx} embeddings")
    embeddings = np.stack(embeddings)
    labels = np.array(labels, dtype=np.int64)

    # ensure output dir exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # save
    logger.info(f"Saving dataset to {output_path}")
    np.savez(output_path,
             embeddings=embeddings,
             labels=labels,
             authors=np.array(author_names, dtype='<U')  # unicode strings
            )
    logger.info(f"Saved {embeddings.shape[0]} samples with dim {embeddings.shape[1]} to {output_path}")
    logger.debug(f"Author names: {author_names}")
    return embeddings, labels, author_names

if __name__ == '__main__':
    build_dataset()
