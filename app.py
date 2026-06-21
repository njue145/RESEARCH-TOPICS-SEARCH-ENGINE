import streamlit as st
import pandas as pd

from utils import (
    load_model,
    generate_embeddings,
    load_embeddings,
    save_embeddings,
    search_topic
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Research Topic Similarity Checker",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "🎓 Research Topic Similarity Checker"
)

st.write(
    """
    Enter a proposed research topic
    and compare it against previously
    completed projects using AI-powered
    semantic similarity.
    """
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:

    df = pd.read_csv(
        "topics.csv", encoding='cp1252'
    )

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "Dataset Information"
)

st.sidebar.metric(
    "Topics",
    len(df)
)

top_n = st.sidebar.slider(
    "Number of Results",
    min_value=5,
    max_value=20,
    value=10
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def get_model():

    return load_model()


model = get_model()

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_data
def get_embeddings():

    cached = load_embeddings()

    if cached is not None:

        return cached

    embeddings = generate_embeddings(
        df,
        model
    )

    save_embeddings(
        embeddings
    )

    return embeddings


embeddings = get_embeddings()

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

topic = st.text_area(
    "Enter Proposed Research Topic"
)

# --------------------------------------------------
# SEARCH BUTTON
# --------------------------------------------------

if st.button(
    "Check Similarity"
):

    if not topic.strip():

        st.warning(
            "Please enter a topic."
        )

    else:

        results = search_topic(
            topic,
            df,
            embeddings,
            model,
            top_n
        )

        st.subheader(
            "Most Similar Topics"
        )

        st.dataframe(
            results[
                [
                    "regno",
                    "name",
                    "Title",
                    "Similarity_%",
                    "Category"
                ]
            ],
            use_container_width=True
        )

        highest = (
            results.iloc[0]
            ["Similarity_%"]
        )

        st.subheader(
            "Interpretation"
        )

        if highest >= 90:

            st.error(
                """
                Potential duplicate topic detected.
                Consider changing the study area,
                methodology, objectives,
                or dataset.
                """
            )

        elif highest >= 70:

            st.warning(
                """
                Significant overlap exists.
                Further differentiation
                is recommended.
                """
            )

        else:

            st.success(
                """
                Topic appears sufficiently
                distinct.
                """
            )

        # --------------------------
        # CHART
        # --------------------------


        # --------------------------
        # DOWNLOAD
        # --------------------------

        csv = results.to_csv(
            index=False
        )

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="similarity_results.csv",
            mime="text/csv"
        )
