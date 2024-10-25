# jp_bert_embedding

## setup
```sh
git clone git@github.com:aqua-pigeon/embed_models.git
cd embed_models
pyenv install 3.8.10
pyenv local 3.8.10 embmodel
pyenv activate jbe
pip install -r requirements.txt
```

## prepare data
1. download stopwords
    ```sh
    wget http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt -O data/stopwords.txt
    ```
    place stopwords.txt at `data/stopwords.txt`

2. download interview history  
place interview history at `data/interview_history.csv`
    <details>
    <summary>sample data</summary>
    ```json
    {
        "len": 1,
        "data": [
            {
                "session_id": "3f5aafa7-0887-48bc-ac6d-01aa68e739be",
                "line_id": "U4b9c5834b9613588cafdfc15344a42d2",
                "data": [
                    {
                        "role": "assistant",
                        "content": "最近あなたが責任感を持って行なったタスクについて具体的に教えてもらえますか？"
                    },
                    {
                        "role": "user",
                        "content": "バイトの業務"
                    },
                    {
                        "role": "assistant",
                        "content": "それは興味深いですね。バイト中に特に意識していることは何ですか？"
                    },
                    {
                        "role": "user",
                        "content": "効率よく物事を進めること"
                    },
                    {
                        "role": "assistant",
                        "content": "それは素晴らしいですね！効率よく進めるために、どんな情報や経験が特に役立っていると感じますか？"
                    },
                    {
                        "role": "user",
                        "content": "仕事内容について予習しています！"
                    },
                    {
                        "role": "assistant",
                        "content": "そうなんですね、すごいです！ドリンク作成で特に大切だと感じるステップはありますか？"
                    },
                    {
                        "role": "user",
                        "content": "私はドリンク作成はしていません！"
                    },
                    {
                        "role": "assistant",
                        "content": "それは興味深いですね。アルバイトで効率よく進めるために、特に心がけていることはありますか？"
                    },
                    {
                        "role": "user",
                        "content": "事前に仕事内容を予習してからバイト先に出勤しました！"
                    },
                    {
                        "role": "assistant",
                        "content": "それは素晴らしいですね！予習した中で、特に役立った情報は何ですか？"
                    },
                    {
                        "role": "user",
                        "content": "おにぎり屋で働いていますが、おにぎりとおかずの詰め方が書かれたマニュアルが役に立ちました。"
                    },
                    {
                        "role": "assistant",
                        "content": "そのマニュアル、役立っているんですね。具体的には、どんなふうに活用していますか？"
                    },
                    {
                        "role": "user",
                        "content": "作り始める前にマニュアルを見るようにしています"
                    },
                    {
                        "role": "assistant",
                        "content": "そうなんですね、マニュアルを使っているんですね。作業中に特に大事にしているポイントはありますか？"
                    }
                ],
                "end": true
            },
        ]
    }
    ```
    </details>