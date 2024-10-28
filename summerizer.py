import json
import openai
import argparse
from dotenv import load_dotenv
import os
import tqdm
import requests


# LLMを用いて要約を生成する関数
def generate_summary_using_llm(interview_data, model_name: str, instruction: str):
    # インタビューデータをテキスト形式に変換
    interview_text = "\n".join(
        [entry["content"] for entry in interview_data if entry["role"] == "user"]
    )

    # OpenAIのAPIを使って要約を生成
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": interview_text},
        ],
    )

    # 生成された要約を返す
    summary = response.choices[0].message.content
    return summary


def translate_text(from_lang, to_lang, text):
    url = "https://script.google.com/macros/s/AKfycbwS07PrEtK3zvoCP9W2gtWuxkW-VFIs3d4JM0K3kA4oQv0IVn_vXm2aQ7xsjpvmrI7jWQ/exec"  # ここにウェブアプリのデプロイURLを入力
    params = {"from": from_lang, "to": to_lang, "text": text}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "translatedText" in data:
            return data["translatedText"]
        else:
            return f"Error: {data.get('error', 'Unknown error')}"
    else:
        return f"HTTP Error: {response.status_code}"


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
    インタビューデータから、{action}{recognition}{infomation}{attitude}の四つの内容を英語で要約して抽出してください。
    ただし、actionは回答者の現在のタスクについての行動内容について、recognitionは認知タスク分析において重要な認知プロセスに関わる判断や思考、気付き、手がかり、目標などであり、
    infomationは現在のタスクにあたって活用している情報や経験についてであり、attitudeは仕事に対する取り組み姿勢や信念です。これらの四つの項目に分類される内容は複数あって構いません。
    名詞形で、言語は**英語**でお願いします。出力形式は次に示すjsonに従うこと．
    { 
        "action": [
            ""
        ],
        "recognition": [
            ""
        ],
        "information": [
            ""
        ],
        "attitude": [
            ""
        ]
    }
    """,
        help="Instruction for the LLM",
    )

    parser.add_argument(
        "--target_data_path",
        type=str,
        default="data/all_messages.json",
        help="Path to the JSON file containing the interview data",
    )
    parser.add_argument(
        "--summerized_file_path",
        type=str,
        default="data/summerized_data.json",
        help="Path to the output file for the summerized data",
    )
    args = parser.parse_args()

    load_dotenv()
    args.openai_api_key = os.getenv("OPENAI_API_KEY")
    args.gemin_api_key = os.getenv("GEMINI_API_KEY")
    args.line_channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    args.mongodb_username = os.getenv("MONGODB_USERNAME")
    args.mongodb_password = os.getenv("MONGODB_PASSWORD")
    return args


if __name__ == "__main__":
    args = arg_parse()

    # OpenAI APIの設定
    openai.api_key = args.openai_api_key

    # JSONファイルを読み込み
    data = json.load(open(args.target_data_path))

    # インタビューデータのみを抽出
    interview_histories = []
    for d in data["data"]:
        interview_histories.append(d)

    summerized_data = []

    error_sum_count = 0

    # セッションごとに要約と特徴量抽出を実施
    for interview_history in tqdm.tqdm(interview_histories, desc="Extracting features"):
        session_id = interview_history["session_id"]
        interview_data = interview_history["data"]

        error_count = 0

        while True:
            # LLMを使って要約を生成
            summary = generate_summary_using_llm(
                interview_data, args.openai_model, args.instruction
            )

            # 要らない文字列を削除, ```json, ```を削除
            summary = summary.replace("```json", "").replace("```", "")
            #結果を翻訳
            # summary = translate_text("en", "ja", summary)

            # 結果を確認
            # print(f"要約結果 (Session ID: {session_id}):\n{summary}")

            try:
                # 要約を文ごとに分割
                sentences = json.loads(summary)

                # check
                # print(sentences)
                # input("Press Enter to continue...")

                summerized_data.append({"session_id": session_id, "summary": sentences})

                error_sum_count += error_count
                break
            except json.JSONDecodeError:
                error_count += 1
                # print(f"Error count: {error_count}")
                if error_count > 5:
                    print("Error count is over 5.")
                    break

    # 要約結果を保存
    json.dump(
        summerized_data,
        open(args.summerized_file_path, "w"),
        ensure_ascii=False,
        indent=4,
    )

    print(f"Error percentage: {error_sum_count / len(interview_histories)}")
