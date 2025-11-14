import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm

# 1. 载入清洗后的数据
df = pd.read_csv("youtube_comments_clean.csv")
df["comment_clean"] = df["comment_clean"].astype(str)
df = df[df["comment_clean"].str.strip().str.len() > 0]


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# 2. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = 0 if torch.cuda.is_available() else -1  # GPU 优先

# 3. 创建 sentiment pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device,            # GPU: 0, CPU: -1
    truncation=True
)

# 4. 为了加速，分 batch 预测
texts = df["comment_clean"].tolist()
batch_size = 32

all_labels = []
all_scores = []

for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
    batch = texts[i:i+batch_size]
    outputs = sentiment_pipeline(batch, truncation=True)
    for out in outputs:
        all_labels.append(out["label"])
        all_scores.append(out["score"])

# 5. 写回 DataFrame
df["sentiment_label"] = all_labels
df["sentiment_score"] = all_scores

# 可选：把 label 转成数值，方便后面统计
label_map = {"negative": -1, "neutral": 0, "positive": 1,
             "NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
df["sentiment_numeric"] = df["sentiment_label"].map(label_map)

df.to_csv("youtube_comments_with_sentiment.csv", index=False)
print(df[["comment", "sentiment_label", "sentiment_score"]].head(10))
