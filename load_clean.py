import pandas as pd
import re
import emoji

# 1. read
df = pd.read_csv("youtube_comments.csv")

# 2. leave non-empty comments
df = df.dropna(subset=["comment"])

# 3. remove url, emoji
def clean_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = emoji.replace_emoji(text, replace="") # remove emoji
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["comment_clean"] = df["comment"].apply(clean_text)
print(df.head())

df.to_csv("youtube_comments_clean.csv", index=False)
