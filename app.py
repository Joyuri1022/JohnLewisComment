import streamlit as st
import pandas as pd

# ===== 基本配置 =====
st.set_page_config(
    page_title="YouTube Sentiment Dashboard",
    layout="wide"
)

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    # 确保这些列存在，如果你的列名不同，这里改一下
    if "comment_clean" not in df.columns and "comment" in df.columns:
        df["comment_clean"] = df["comment"].astype(str)
    # 统一类型
    df["comment_clean"] = df["comment_clean"].astype(str)
    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].astype(str)
    return df


# ===== 侧边栏：数据加载 & 过滤条件 =====
st.sidebar.title("Settings")

csv_path = st.sidebar.text_input(
    "CSV file path",
    value="youtube_comments_with_sentiment.csv"
)

if not csv_path:
    st.sidebar.warning("请输入 CSV 路径")
    st.stop()

try:
    df = load_data(csv_path)
except Exception as e:
    st.sidebar.error(f"无法加载 CSV：{e}")
    st.stop()

st.sidebar.success(f"已加载 {len(df)} 条评论")

# 情绪筛选
if "sentiment_label" in df.columns:
    all_sentiments = sorted(df["sentiment_label"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Filter by sentiment",
        options=all_sentiments,
        default=all_sentiments
    )
else:
    selected_sentiments = None

# 关键词搜索
keyword = st.sidebar.text_input("Keyword in comment (optional)", value="")

# 最小点赞
if "likes" in df.columns:
    min_likes = int(df["likes"].min())
    max_likes = int(df["likes"].max())
    like_filter = st.sidebar.slider(
        "Minimum likes",
        min_value=min_likes,
        max_value=max_likes,
        value=min_likes
    )
else:
    like_filter = None

# ===== 应用过滤 =====
filtered = df.copy()

if selected_sentiments is not None and len(selected_sentiments) > 0:
    filtered = filtered[filtered["sentiment_label"].isin(selected_sentiments)]

if keyword.strip():
    kw = keyword.lower().strip()
    filtered = filtered[filtered["comment_clean"].str.lower().str.contains(kw)]

if like_filter is not None and "likes" in filtered.columns:
    filtered = filtered[filtered["likes"] >= like_filter]

# ===== 主界面：标题 =====
st.title("YouTube Comments Sentiment Dashboard")

st.markdown(
    "这个面板展示了你对某个 YouTube 视频评论做的情感分析结果。"
)

# ===== 全局指标 =====
col1, col2, col3 = st.columns(3)

total_comments = len(df)
filtered_comments = len(filtered)

with col1:
    st.metric("Total comments", total_comments)
with col2:
    st.metric("Filtered comments", filtered_comments)
with col3:
    if "sentiment_label" in filtered.columns:
        pos_ratio = (filtered["sentiment_label"].str.lower() == "positive").mean()
        st.metric("Positive ratio (filtered)", f"{pos_ratio*100:.1f}%")

st.divider()

# ===== 情绪分布图 =====
if "sentiment_label" in filtered.columns:
    st.subheader("Sentiment distribution (filtered)")
    sent_counts = (
        filtered["sentiment_label"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    st.bar_chart(sent_counts.set_index("sentiment"))

    st.dataframe(sent_counts, use_container_width=True)
else:
    st.info("当前数据没有 sentiment_label 列，无法画情绪分布。")

st.divider()

# ===== 时间维度（如果有 publishedAt） =====
if "publishedAt" in filtered.columns:
    st.subheader("Sentiment over time")

    # 转成 datetime
    filtered["_published_dt"] = pd.to_datetime(filtered["publishedAt"], errors="coerce")
    time_df = filtered.dropna(subset=["_published_dt"]).copy()
    if not time_df.empty:
        # 按日期聚合
        time_df["date"] = time_df["_published_dt"].dt.date
        daily = (
            time_df
            .groupby(["date", "sentiment_label"])
            .size()
            .rename("count")
            .reset_index()
        )

        # 转为透视表便于画图
        pivot = daily.pivot(index="date", columns="sentiment_label", values="count").fillna(0)
        st.line_chart(pivot)
    else:
        st.info("publishedAt 列无法转换为日期。")

st.divider()

# ===== 展示具体评论（表格） =====
st.subheader("Sample comments (filtered)")

n_show = st.slider("Number of rows to show", min_value=5, max_value=200, value=50)

cols_to_show = []
for col in ["comment", "comment_clean", "sentiment_label", "sentiment_score", "likes", "publishedAt"]:
    if col in filtered.columns:
        cols_to_show.append(col)

st.dataframe(
    filtered[cols_to_show].head(n_show),
    use_container_width=True
)
