import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import MeCab

class embedding:
    def __init__(self):
        self.sentence_transformers = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        self.mecab = MeCab.Tagger()

    def get_sentences_embed(self, sentences:list)->list:
        if type(sentences) is str:
            sentences = [sentences]
        embeddings:list = self.sentence_transformers.encode(sentences)
        return embeddings
    
    def get_tf_idf_embed(self, sentences:list)->list:
        # vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b', ngram_range=(1, 2))
        # tfidf_matrix = vectorizer.fit_transform(sentences)
        # return tfidf_matrix.toarray().tolist()
        pass

    def morphological_analysis(self, sentences:list)->list:
        if type(sentences) is str:
            sentences = [sentences]
        words = []
        wakati = MeCab.Tagger("-Owakati")
        for sentence in sentences:
            word = wakati.parse(sentence).split()

            words.append(word)
        return words
    
    def remove_stop_words(self, sentences:list)->list:
        stopwords_path
        

if __name__ == '__main__':
    emb = embedding()
    
    sentences = ['警告表示を見つけた際には、安全マニュアルに従って行動する']
    
    # # bert特徴量抽出
    embeddings = emb.get_sentences_embed(sentences)
    print(embeddings)
    
    # 形態素解析
    # words = emb.morphological_analysis(sentences)
    # print(words)