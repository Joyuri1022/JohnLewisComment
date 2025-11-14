from googleapiclient.discovery import build
import pandas as pd
from tqdm import tqdm
import os
API_KEY = os.getenv("YT_API_KEY")
VIDEO_ID = "z1bRlnyQeDk"

def get_youtube_comments(video_id, api_key):
    youtube = build("youtube", "v3", developerKey=api_key)

    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": snippet.get("authorDisplayName"),
                "comment": snippet.get("textOriginal"),
                "likes": snippet.get("likeCount"),
                "publishedAt": snippet.get("publishedAt")
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


if __name__ == "__main__":
    comments = get_youtube_comments(VIDEO_ID, API_KEY)
    df = pd.DataFrame(comments)
    df.to_csv("youtube_comments.csv", index=False)
    print(f"Saved {len(df)} comments to youtube_comments.csv")
