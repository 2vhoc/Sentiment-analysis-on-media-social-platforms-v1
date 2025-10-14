import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from crawl_data.crawl_cmt_from_ytb import Crawler
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

def process(comments):
    with open('notebook/models/tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    sequences = tokenizer.texts_to_sequences(comments)
    pad = pad_sequences(sequences, padding='post', maxlen=30, truncating='post')
    model = load_model('notebook/models/model.h5')
    predictions = model.predict(pad)
    return np.argmax(predictions, axis=1)

def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="▶️")
    st.title("▶️ Sentiment Analysis")
    st.markdown("---")

    youtube_url = st.text_input(
        "Nhập link (URL) của video YouTube vào ô dưới đây:",
        placeholder="Ví dụ: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    st.subheader("Trạng thái và Kết quả")
    if youtube_url:
        st.success(f"Đã nhận được link: **{youtube_url}**")

        if st.button("Bắt đầu xử lý link"):
            with st.spinner("Đang chạy logic xử lý và cào dữ liệu... Vui lòng chờ!"):
                try:
                    cmts = Crawler(youtube_url)
                    cmts.get_youtube_comments()

                    comments = [c['text'] for c in cmts.comments if isinstance(c, dict) and 'text' in c]
                    authors = [c['author'] for c in cmts.comments if isinstance(c, dict) and 'author' in c]

                    if not comments:
                        st.warning("Không có comment hợp lệ để phân tích!")
                        return

                    result = process(comments)
                    sentiment = {2: "positive", 1: "negative", 0: "neutral"}

                    comments_by_sentiment = {"positive": [], "neutral": [], "negative": []}
                    for c, r in zip(comments, result):
                        label = sentiment[int(r)]
                        comments_by_sentiment[label].append(c)

                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                    counts = {k: len(v) for k, v in comments_by_sentiment.items()}
                    axes[0].bar(counts.keys(), counts.values(), color=['green', 'gray', 'red'], label='Số lượng')
                    axes[0].set_title("Số lượng comment theo sentiment")
                    axes[0].set_ylabel("Số lượng comment")
                    axes[0].legend()

                    avg_lengths = {k: (np.mean([len(x) for x in v]) if v else 0) for k, v in comments_by_sentiment.items()}
                    axes[1].bar(avg_lengths.keys(), avg_lengths.values(), color=['green', 'gray', 'red'], label='Độ dài trung bình')
                    axes[1].set_title("Độ dài trung bình comment theo sentiment")
                    axes[1].set_ylabel("Số ký tự trung bình")
                    axes[1].legend()

                    top_authors = Counter(authors).most_common(5)
                    names, values = zip(*top_authors) if top_authors else ([], [])
                    axes[2].bar(names, values, color='blue', label='Số comment')
                    axes[2].set_title("Top 5 tác giả comment nhiều nhất")
                    axes[2].set_ylabel("Số lượng comment")
                    axes[2].legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.success("Xử lý hoàn tất 🎉")

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

    else:
        st.info("Vui lòng nhập một link YouTube để bắt đầu.")

    st.markdown("---")
    st.caption("Ứng dụng đơn giản được xây dựng bằng Streamlit.")

if __name__ == "__main__":
    main()
