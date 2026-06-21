import re
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILE = "embedding_cache.pkl"


def clean_text(text):
    """
    Clean research title.
    """

    text = str(text).lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


def load_model():
    """
    Load transformer model.
    """

    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    df,
    model
):
    """
    Generate embeddings.
    """

    titles = df["Title"].apply(
        clean_text
    ).tolist()

    embeddings = model.encode(
        titles,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings


def save_embeddings(
    embeddings
):

    with open(
        CACHE_FILE,
        "wb"
    ) as f:

        pickle.dump(
            embeddings,
            f
        )


def load_embeddings():

    if Path(
        CACHE_FILE
    ).exists():

        with open(
            CACHE_FILE,
            "rb"
        ) as f:

            return pickle.load(f)

    return None


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

    else:
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
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    results = df.copy()

    results["Similarity"] = similarities

    results["Similarity_%"] = (
        similarities * 100
    ).round(2)

    results["Category"] = (
        results["Similarity"]
        .apply(classify_similarity)
    )

    results = results.sort_values(
        by="Similarity",
        ascending=False
    )

    return results.head(top_n)