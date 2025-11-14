import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm

# 1. load clean data
df = pd.read_csv("youtube_comments_clean.csv")
df["comment_clean"] = df["comment_clean"].astype(str)
df = df[df["comment_clean"].str.strip().str.len() > 0]

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# 2. tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = 0 if torch.cuda.is_available() else -1  # GPU

# 3. emotion pipeline
# text-classification
sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,    # GPU: 0, CPU: -1
    truncation=True
)

# 4. batch
texts = df["comment_clean"].tolist()
batch_size = 32

all_labels = []
all_scores = []

for i in tqdm(range(0, len(texts), batch_size), desc="Emotion"):
    batch = texts[i:i+batch_size]
    outputs = sentiment_pipeline(batch, truncation=True)
    for out in outputs:
        all_labels.append(out["label"])
        all_scores.append(out["score"])

# 5. write DataFrame
df["sentiment_label"] = all_labels
df["sentiment_score"] = all_scores

# anger=0, disgust=1, fear=2, joy=3, sadness=4, surprise=5
df["sentiment_numeric"] = pd.Categorical(df["sentiment_label"]).codes

df.to_csv("youtube_comments_with_emotion.csv", index=False)

cols_to_show = [c for c in ["comment", "comment_clean", "sentiment_label", "sentiment_score", "sentiment_numeric"] if c in df.columns]
print(df[cols_to_show].head(10))
