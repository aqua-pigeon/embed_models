import json
import openai
from sentence_transformers import SentenceTransformer
import argparse
from dotenv import load_dotenv
import os
import tqdm


class Embedding:
    def __init__(self, model_name: str):
        self.sentence_transformers = SentenceTransformer(model_name)

    def get_sentences_embed(self, sentences: list) -> list:
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.sentence_transformers.encode(sentences)
        return embeddings


def generate_summary_using_llm(interview_data, model_name: str, instruction: str):
    interview_text = "\n".join(
        [entry["content"] for entry in interview_data if entry["role"] == "user"]
    )

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": interview_text},
        ],
    )
    summary = response.choices[0].message.content
    return summary


def arg_parse():
    parser = argparse.ArgumentParser(description="Embedding")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--openai_model", type=str, default="gpt-4o", help="OpenAI model name"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="""
    インタビューデータから、{action}{recognition}{infomation}{attitude}の四つの内容を要約して抽出してください。
    ただし、actionは回答者の現在のタスクについての行動内容について、recognitionは認知タスク分析において重要な認知プロセスに関わる判断や思考、気付き、手がかり、目標などであり、
    infomationは現在のタスクにあたって活用している情報や経験についてであり、attitudeは仕事に対する取り組み姿勢や信念です。これらの四つの項目に分類される内容は複数あって構いません。
    動詞は名詞形にしてください。ただし、要約しても具体的な回答者の発言内容が失われないようにしてください。例えば、「過去の経験」という表現は抽象的なので、インタビューに情報があれば具体的にその経験内容も含んでください。
    """,
        help="Instruction for the LLM",
    )
    parser.add_argument(
        "--target_data_path",
        type=str,
        default="data/all_messages.json",
        help="Path to the JSON file containing the interview data",
    )
    args = parser.parse_args()

    load_dotenv()
    args.openai_api_key = os.getenv("OPENAI_API_KEY")
    return args


def calculate_sse(embeddings, max_clusters=15):
    sse = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)
    return sse


def plot_sse(sse):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sse) + 1), sse, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE (Sum of Squared Errors)")
    plt.title("Elbow Method for Optimal k")
    plt.show()


if __name__ == "__main__":
    args = arg_parse()
    emb = Embedding(args.model_name)
    openai.api_key = args.openai_api_key

    # JSONファイルを読み込み
    data = json.load(open(args.target_data_path))

    # インタビューデータを収集
    interview_histories = []
    for d in data["data"]:
        interview_histories.append(d)

    features = {"action": [], "recognition": [], "information": [], "attitude": []}
    text_pairs = {"action": [], "recognition": [], "information": [], "attitude": []}

    for interview_history in tqdm.tqdm(interview_histories, desc="Extracting features"):
        session_id = interview_history["session_id"]
        interview_data = interview_history["data"]

        # LLMを使って要約を生成
        summary = generate_summary_using_llm(
            interview_data, args.openai_model, args.instruction
        )

        summary_lines = summary.split("\n")
        for line in summary_lines:
            if "action:" in line:
                features["action"].append(emb.get_sentences_embed(line))
                text_pairs["action"].append({"session_id": session_id, "text": line})
            elif "recognition:" in line:
                features["recognition"].append(emb.get_sentences_embed(line))
                text_pairs["recognition"].append(
                    {"session_id": session_id, "text": line}
                )
            elif "information:" in line:
                features["information"].append(emb.get_sentences_embed(line))
                text_pairs["information"].append(
                    {"session_id": session_id, "text": line}
                )
            elif "attitude:" in line:
                features["attitude"].append(emb.get_sentences_embed(line))
                text_pairs["attitude"].append({"session_id": session_id, "text": line})

    num_clusters = 5
    cluster_results = {}

    for category, embeddings in features.items():
        if embeddings:
            embeddings = np.array(embeddings).squeeze()

            # SSE計算とプロット
            sse = calculate_sse(embeddings, max_clusters=15)
            plot_sse(sse)

            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            labels = kmeans.fit_predict(embeddings)

            cluster_results[category] = []
            for i, label in enumerate(labels):
                cluster_results[category].append(
                    {
                        "cluster": label,
                        "session_id": text_pairs[category][i]["session_id"],
                        "text": text_pairs[category][i]["text"],
                    }
                )

    for category, results in cluster_results.items():
        print(f"\nクラスタリング結果 - {category}:")
        for result in results:
            print(
                f"  クラスター: {result['cluster']}, セッションID: {result['session_id']}, 要約: {result['text']}"
            )
