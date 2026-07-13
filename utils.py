import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILE = "embedding_cache.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_model():
    """
    Load Sentence Transformer model.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )


def generate_embeddings(df, model):
    """
    Used ONLY when creating embedding_cache.pkl locally.
    """

    titles = df["Title"].apply(clean_text).tolist()

    embeddings = model.encode(
        titles,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )

    return embeddings


def save_embeddings(embeddings):

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings():

    if not Path(CACHE_FILE).exists():
        return None

    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def classify_similarity(score):

    percentage = score * 100

    if percentage >= 90:
        return "Potential Duplicate"

    elif percentage >= 85:
        return "Very High Similarity"

    elif percentage >= 70:
        return "Moderate Similarity"

    elif percentage >= 50:
        return "Low Similarity"

    return "Distinct"


def search_topic(
    topic,
    df,
    embeddings,
    model,
    top_n=10
):

    query = clean_text(topic)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    results = df.copy()

    results["Similarity"] = similarities
    results["Similarity_%"] = (similarities * 100).round(2)
    results["Category"] = results["Similarity"].apply(classify_similarity)

    return results.sort_values(
        by="Similarity",
        ascending=False
    ).head(top_n)
