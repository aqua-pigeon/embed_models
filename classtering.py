import json
from sentence_transformers import SentenceTransformer
import argparse
import tqdm
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import openai
from dotenv import load_dotenv


class Embedding:
    def __init__(self, model_name: str):
        self.sentence_transformers = SentenceTransformer(model_name)

    def get_sentences_embed(self, sentences: list) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.sentence_transformers.encode(sentences)
        return embeddings

    def get_sentences_embed_from_json(self, json_file_path: str) -> list:
        data_list = json.load(open(json_file_path, "r"))
        for data in tqdm.tqdm(data_list, desc="Extracting features"):
            summary_data = data["summary"]

            action = summary_data["action"]
            recognition = summary_data["recognition"]
            information = summary_data["information"]
            attitude = summary_data["attitude"]

            action_embeddings = self.get_sentences_embed(action).tolist()
            recognition_embeddings = self.get_sentences_embed(recognition).tolist()
            information_embeddings = self.get_sentences_embed(information).tolist()
            attitude_embeddings = self.get_sentences_embed(attitude).tolist()

            data["embeddings"] = {
                "action": action_embeddings,
                "recognition": recognition_embeddings,
                "information": information_embeddings,
                "attitude": attitude_embeddings,
            }
        return data_list


def cluster_embeddings(data_list, num_clusters):
    cluster_results = {}

    # 各カテゴリごとにクラスタリングを行う
    for category in ["action", "recognition", "information", "attitude"]:
        embeddings = []
        references = []  # 各embeddingのsession_idと要約を保持

        # カテゴリ内のすべてのembeddingを収集
        for data in data_list:
            session_id = data["session_id"]
            for emb, text in zip(
                data["embeddings"][category], data["summary"][category]
            ):
                embeddings.append(emb)
                references.append({"session_id": session_id, "text": text})

        embeddings = np.array(embeddings)

        # クラスタリング
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # クラスタ結果を保存
        cluster_results[category] = []
        for i, label in enumerate(labels):
            cluster_results[category].append(
                {
                    "cluster": int(label),
                    "session_id": references[i]["session_id"],
                    "text": references[i]["text"],
                }
            )

    return cluster_results


def calculate_sse(embeded_data, max_clusters):
    action_embeddings = []
    recognition_embeddings = []
    information_embeddings = []
    attitude_embeddings = []

    for data in embeded_data:
        action_embeddings.extend(data["embeddings"]["action"])
        recognition_embeddings.extend(data["embeddings"]["recognition"])
        information_embeddings.extend(data["embeddings"]["information"])
        attitude_embeddings.extend(data["embeddings"]["attitude"])

    action_embeddings = np.array(action_embeddings)
    recognition_embeddings = np.array(recognition_embeddings)
    information_embeddings = np.array(information_embeddings)
    attitude_embeddings = np.array(attitude_embeddings)

    action_sse = []
    recognition_sse = []
    information_sse = []
    attitude_sse = []

    for i in tqdm.tqdm(range(1, max_clusters + 1), desc="Calculating SSE"):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
        action_sse.append(kmeans.fit(action_embeddings).inertia_)
        recognition_sse.append(kmeans.fit(recognition_embeddings).inertia_)
        information_sse.append(kmeans.fit(information_embeddings).inertia_)
        attitude_sse.append(kmeans.fit(attitude_embeddings).inertia_)
    return action_sse, recognition_sse, information_sse, attitude_sse


def plot_sse(sse, file_name: str):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sse) + 1), sse, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE (Sum of Squared Errors)")
    plt.title("Elbow Method for Optimal k")
    plt.savefig(file_name)


# GPT-4 APIを使って上位概念を生成する関数
def generate_abstract_concept(texts, model_name="gpt-4o"):
    error_count = 0

    while True:
        prompt = (
            f"次の内容は同じクラスターに属する関連する要素です。これらを上位概念のラベルで表すとどのような内容になりますか？{texts}.出力は次の形式に従うこと。"
            + """{"result": ""}"""
        )
        response = openai.ChatCompletion.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        # GPT-4からの返答を抽出
        response_text = response.choices[0].message.content.strip()
        response_text = (
            response_text.replace("Output:", "")
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        try:
            result = json.loads(response_text)["result"]
            return result
        except json.JSONDecodeError:
            error_count += 1
            if error_count > 3:
                print(f"Error: gpt-4o response is not JSON format: {response_text}")
                return None


def generate_cluster_concepts(category_data: list) -> dict:
    cluster_texts = {}  # key = category_data.cluster, value = [category_data.text]
    cluster_concepts = {}  # key = category_data.cluster, value = abstract concept

    # cluserごとに単語を集約
    for data in category_data:
        cluster_id = data["cluster"]
        text = data["text"]
        if cluster_id not in cluster_texts:
            cluster_texts[cluster_id] = []
        cluster_texts[cluster_id].append(text)

    # clusterごとに上位概念を生成
    for cluster_id, texts in cluster_texts.items():
        cluster_concepts[cluster_id] = generate_abstract_concept(texts)

    return cluster_concepts


def arg_parse():
    parser = argparse.ArgumentParser(description="Embedding")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--summerized_file_path",
        type=str,
        default="data/summerized_data.json",
        help="Path to the JSON file containing the interview data",
    )
    parser.add_argument(
        "--feature_data_path",
        type=str,
        default="data/feature_data.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=15,
        help="Number of clusters",
    )
    parser.add_argument(
        "--classtered_data_path",
        type=str,
        default="data/classtered_data.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--calc_sse",
        action="store_true",
        help="Whether to calculate the SSE",
    )
    parser.add_argument(
        "--plot_file_dir",
        type=str,
        default="data/elbow_fig",
        help="Path to the output plot file",
    )
    parser.add_argument(
        "--concept_file_path",
        type=str,
        default="data/concepts.json",
        help="Path to the output JSON file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    # クラスターごとに上位概念を生成する関数
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    emb = Embedding(args.model_name)

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(args.plot_file_dir):
        os.makedirs(args.plot_file_dir)

    # embeddingを取得
    if os.path.exists(args.feature_data_path):
        embeded_data = json.load(open(args.feature_data_path, "r"))
    else:
        embeded_data = emb.get_sentences_embed_from_json(args.summerized_file_path)
        json.dump(
            embeded_data,
            open(args.feature_data_path, "w"),
            indent=4,
            ensure_ascii=False,
        )

    # sseを計算
    if args.calc_sse:
        action_sse, recognition_sse, information_sse, attitude_sse = calculate_sse(
            embeded_data, max_clusters=args.max_clusters
        )
        plot_sse(action_sse, os.path.join(args.plot_file_dir, "action_sse.png"))
        plot_sse(
            recognition_sse, os.path.join(args.plot_file_dir, "recognition_sse.png")
        )
        plot_sse(
            information_sse, os.path.join(args.plot_file_dir, "information_sse.png")
        )
        plot_sse(attitude_sse, os.path.join(args.plot_file_dir, "attitude_sse.png"))

    # clustering
    if os.path.exists(args.classtered_data_path):
        cluster_results = json.load(open(args.classtered_data_path, "r"))
    else:
        cluster_results = cluster_embeddings(embeded_data, num_clusters=20)
        json.dump(
            cluster_results,
            open("data/cluster_results.json", "w"),
            indent=4,
            ensure_ascii=False,
        )

    # 上位概念を生成
    if os.path.exists(args.concept_file_path):
        concept_results = json.load(open(args.concept_file_path, "r"))
    else:
        category_list = ["action", "recognition", "information", "attitude"]
        concept_results = {}
        for category in tqdm.tqdm(category_list, desc="Generating Concepts"):
            cluster_concepts = generate_cluster_concepts(cluster_results[category])
            concept_results[category] = cluster_concepts
        json.dump(
            concept_results,
            open(args.concept_file_path, "w"),
            indent=4,
            ensure_ascii=False,
        )
