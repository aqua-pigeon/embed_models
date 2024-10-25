import json
import openai
from sentence_transformers import SentenceTransformer
import argparse
from dotenv import load_dotenv
import os


class Embedding:
    def __init__(self):
        self.sentence_transformers = SentenceTransformer(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
        )

    def get_sentences_embed(self, sentences: list) -> list:
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.sentence_transformers.encode(sentences)
        return embeddings


# LLMを用いて要約を生成する関数
def generate_summary_using_llm(interview_data):
    # 指示文
    instruction = """
    インタビューデータから、{action}{recognition}{infomation}{attitude}の四つの内容を要約して抽出してください。
    ただし、actionは回答者の現在のタスクについての行動内容について、recognitionは認知タスク分析において重要な認知プロセスに関わる判断や思考、気付き、手がかり、目標などであり、
    infomationは現在のタスクにあたって活用している情報や経験についてであり、attitudeは仕事に対する取り組み姿勢や信念です。これらの四つの項目に分類される内容は複数あって構いません。
    動詞は名詞形にしてください。
    """

    # インタビューデータをテキスト形式に変換
    interview_text = "\n".join(
        [entry["content"] for entry in interview_data if entry["role"] == "user"]
    )

    # OpenAIのAPIを使って要約を生成
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": interview_text},
        ],
    )

    # 生成された要約を返す
    summary = response.choices[0].message.content
    return summary


def arg_parse():
    parser = argparse.ArgumentParser(description="Embedding")
    args = parser.parse_args()

    load_dotenv()
    args.openai_api_key = os.getenv("OPENAI_API_KEY")
    args.gemin_api_key = os.getenv("GEMINI_API_KEY")
    args.line_channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    args.mongodb_username = os.getenv("MONGODB_USERNAME")
    args.mongodb_password = os.getenv("MONGODB_PASSWORD")
    return args


if __name__ == "__main__":
    emb = Embedding()
    args = arg_parse()

    # OpenAI APIの設定
    openai.api_key = args.openai_api_key

    # JSONファイルを読み込み
    with open("all_messages (1).json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # 複数のインタビューセッションを処理
    interview_histories = []
    for d in data["data"]:
        interview_histories.append(d)

    # セッションごとに要約と特徴量抽出を実施
    for interview_history in interview_histories:
        session_id = interview_history["session_id"]
        interview_data = interview_history["data"]

        # LLMを使って要約を生成
        summary = generate_summary_using_llm(interview_data)
        print(f"要約結果 (Session ID: {session_id}):\n{summary}")

        # 要約を文ごとに分割
        sentences = summary.split("\n")  # 改行で文を分割

        # 特徴量抽出
        if sentences:
            embeddings = emb.get_sentences_embed(sentences)
            print(f"特徴量抽出結果 (Session ID: {session_id}):")
            print(embeddings)
        else:
            print(f"No valid content for session: {session_id}")
