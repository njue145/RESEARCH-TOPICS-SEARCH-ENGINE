```python
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    load_model,
    generate_embeddings,
    load_embeddings,
    save_embeddings,
    search_topic
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Research Topic Similarity Checker",
    page_icon="🎓",
    layout="wide"
)

# ==================================================
# CUSTOM HEADER
# ==================================================

st.markdown("""
<div style='text-align:center;
padding:20px;
border-radius:10px;
background-color:#F8F9FA;
margin-bottom:20px;'>

<h1>🎓 Research Topic Similarity Checker</h1>

<p style='font-size:18px;'>
AI-powered semantic similarity analysis for detecting duplicate
or highly related student research projects.
</p>

</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

try:

    df = pd.read_csv(
        "topics.csv",
        encoding="cp1252"
    )

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()

# ==================================================
# TOP DASHBOARD METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Topics",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Registered Students",
        f"{df['regno'].nunique():,}"
    )

with col3:
    st.metric(
        "AI Model",
        "Sentence-BERT"
    )

st.markdown("---")

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.info(
        """
        Research Topic Search Engine

        This tool assists lecturers in identifying
        duplicate or highly similar research topics.
        """
    )

    st.metric(
        "Topics Available",
        len(df)
    )

    top_n = st.slider(
        "Number of Similar Results",
        min_value=5,
        max_value=20,
        value=10
    )

# ==================================================
# MODEL
# ==================================================

@st.cache_resource
def get_model():

    return load_model()

model = get_model()

# ==================================================
# EMBEDDINGS
# ==================================================

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

# ==================================================
# SEARCH AREA
# ==================================================

with st.container(border=True):

    st.subheader(
        "🔍 Topic Similarity Search"
    )

    topic = st.text_area(
        "Enter Proposed Research Topic",
        height=120,
        placeholder="""
Example:
Assessment of Machine Learning Techniques
for Land Cover Classification Using Sentinel-2 Imagery
"""
    )

    search_clicked = st.button(
        "🚀 Analyze Topic",
        use_container_width=True,
        type="primary"
    )

# ==================================================
# SEARCH
# ==================================================

if search_clicked:

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        with st.spinner(
            "Analyzing similarity..."
        ):

            results = search_topic(
                topic,
                df,
                embeddings,
                model,
                top_n
            )

        best_match = results.iloc[0]
        highest = best_match["Similarity_%"]

        # ==========================================
        # BEST MATCH CARD
        # ==========================================

        st.markdown("---")

        st.subheader(
            "🏆 Closest Match Found"
        )

        st.info(
            f"""
            **Student:** {best_match['name']}

            **Registration Number:** {best_match['regno']}

            **Research Topic:** {best_match['Title']}

            **Similarity Score:** {highest:.2f}%
            """
        )

        # ==========================================
        # RECOMMENDATION
        # ==========================================

        st.subheader(
            "🧠 Recommendation"
        )

        if highest >= 90:

            st.error(
                """
                HIGH RISK:
                This topic is extremely similar to an
                existing project and may require major revision.
                """
            )

        elif highest >= 70:

            st.warning(
                """
                MODERATE RISK:
                Significant overlap exists.
                Review objectives, methodology,
                study area, and datasets.
                """
            )

        else:

            st.success(
                """
                LOW RISK:
                The topic appears sufficiently distinct.
                """
            )

        # ==========================================
        # RESULTS TABLE
        # ==========================================

        st.subheader(
            "📋 Most Similar Topics"
        )

        display_df = results[
            [
                "regno",
                "name",
                "Title",
                "Similarity_%",
                "Category"
            ]
        ].copy()

        display_df["Similarity_%"] = (
            display_df["Similarity_%"]
            .round(2)
            .astype(str) + "%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ==========================================
        # CHART
        # ==========================================

        st.subheader(
            "📊 Similarity Distribution"
        )

        chart_df = results.head(10).copy()

        fig = px.bar(
            chart_df,
            x="Similarity_%",
            y="Title",
            orientation="h",
            text="Similarity_%",
            height=500
        )

        fig.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            },
            xaxis_title="Similarity (%)",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # DOWNLOAD BUTTON
        # ==========================================

        st.subheader(
            "⬇️ Export Results"
        )

        csv = results.to_csv(
            index=False
        )

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="similarity_results.csv",
            mime="text/csv"
        )

# ==================================================
# REPOSITORY PREVIEW
# ==================================================

with st.expander(
    "📚 View Research Repository"
):

    st.dataframe(
        df[
            [
                "regno",
                "name",
                "Title"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    """
    Research Topic Similarity Checker v1.0

    Powered by Streamlit, Sentence Transformers and Scikit-Learn
    """
)
```
