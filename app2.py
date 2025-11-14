import streamlit as st
import pandas as pd
import altair as alt

# Configure Streamlit page
st.set_page_config(page_title="YouTube Emotion Analysis", layout="wide")

@st.cache_data
def load_data(path: str = "youtube_comments_with_emotion.csv") -> pd.DataFrame:
    """
    Load the processed CSV file containing emotion classification results.
    Expected columns:
        - comment_clean
        - sentiment_label
        - sentiment_score
        - sentiment_numeric (optional)
    """
    df = pd.read_csv(path)

    # Ensure text column exists
    if "comment_clean" not in df.columns:
        if "comment" in df.columns:
            df["comment_clean"] = df["comment"].astype(str)
        else:
            df["comment_clean"] = ""

    df["sentiment_label"] = df["sentiment_label"].astype(str)
    df["sentiment_score"] = df["sentiment_score"].astype(float)
    return df


# Load data
df = load_data()

# Title
st.title("📊 John Lewis Christmas Ad Comment Emotion Analysis")

st.markdown(
    """
This dashboard uses the **`j-hartmann/emotion-english-distilroberta-base`** model  
to classify comments into the following 6 emotion categories:

- 😄 **joy**  
- 😢 **sadness**  
- 😡 **anger**  
- 😱 **fear**  
- 🤢 **disgust**  
- 😮 **surprise**
    """
)

# 1. Emotion distribution
st.subheader("1️⃣ Emotion Distribution")

# Fix: avoid duplicate column names
count_df = (
    df["sentiment_label"]
    .value_counts()
    .reset_index()
)
count_df.columns = ["emotion", "count"]  # ensure uniqueness

chart = (
    alt.Chart(count_df)
    .mark_bar()
    .encode(
        x=alt.X("emotion:N", title="Emotion"),
        y=alt.Y("count:Q", title="Number of Comments"),
        tooltip=["emotion", "count"],
    )
)

st.altair_chart(chart, use_container_width=True)

# 2. View comments by emotion
st.subheader("2️⃣ Browse Comments by Emotion")

emotion_options = sorted(df["sentiment_label"].unique().tolist())
selected_emotion = st.selectbox("Select an emotion:", options=emotion_options)

# Filter by emotion
filtered = df[df["sentiment_label"] == selected_emotion].copy()

st.write(f"Total **{len(filtered)}** comments classified as: **{selected_emotion}**")

# Optional keyword search
keyword = st.text_input("Optional keyword filter:", value="")
if keyword.strip():
    filtered = filtered[filtered["comment_clean"].str.contains(keyword, case=False, na=False)]

sort_columns = []
if "publishedAt" in df.columns:
    sort_columns.append("publishedAt")
if "likeCount" in df.columns:
    sort_columns.append("likeCount")

sort_by = st.selectbox("Sort by:", options=sort_columns)

sort_order = st.radio("Order:", ["Ascending", "Descending"], index=1)

if sort_by in filtered.columns:
    filtered = filtered.sort_values(
        by=sort_by,
        ascending=(sort_order == "Ascending")
    )

# Limit displayed rows
top_n = st.slider("Show top N comments:", min_value=5, max_value=200, value=20, step=5)

show_cols = [c for c in ["comment", "comment_clean", "publishedAt", "likeCount", "sentiment_label", "sentiment_score"] if c in filtered.columns]
st.dataframe(filtered[show_cols].head(top_n))


# 3. Emotion score distribution
st.subheader("3️⃣ Confidence Score Distribution")

if not filtered.empty:
    score_chart = (
        alt.Chart(filtered)
        .transform_bin("score_bin", field="sentiment_score", bin=alt.Bin(maxbins=20))
        .mark_bar()
        .encode(
            x=alt.X("score_bin:Q", title="Score"),
            y=alt.Y("count()", title="Number of Comments"),
            tooltip=["count()"],
        )
    )
    st.altair_chart(score_chart, use_container_width=True)
else:
    st.info("No comments match the current filter.")

# 4. Emotion trend over time (line chart using publishedAt)
st.subheader("4️⃣ Emotion Trend Over Time")

# Ensure publishedAt exists and is datetime
if "publishedAt" not in df.columns:
    st.error("Your dataset does not contain a 'publishedAt' column. Please provide a datetime column.")
else:
    # Convert publishedAt to datetime
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    # Drop invalid timestamps
    df_time = df.dropna(subset=["publishedAt"]).copy()

    # Group by date + emotion (daily frequency)
    trend_df = (
        df_time.groupby([pd.Grouper(key="publishedAt", freq="D"), "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    # Line chart
    line_chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("publishedAt:T", title="Date"),
            y=alt.Y("count:Q", title="Number of Comments"),
            color=alt.Color("sentiment_label:N", title="Emotion"),
            tooltip=["publishedAt:T", "sentiment_label:N", "count:Q"]
        )
        .properties(height=400)
    )

    st.altair_chart(line_chart, use_container_width=True)

st.markdown("---")
st.caption("Data source: YouTube comment dataset with emotion classification")
