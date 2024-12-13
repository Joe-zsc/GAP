from sentence_transformers import SentenceTransformer
import os, sys
from pathlib import Path
import torch

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import Configure, UTIL
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import random
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Encoder():

    def __init__(self,
                 pretrained_model_path,
                 SBERT_model_name='',
                 BERT_model_name='',
                 normalization=True):
        pretrained_model_path = Path(pretrained_model_path)
        self.pretrained_model_path = pretrained_model_path
        self.SBERT_model_name = SBERT_model_name
        self.BERT_model_name = BERT_model_name
        if SBERT_model_name:
            self.SBERT_model = SentenceTransformer(
                str(pretrained_model_path / SBERT_model_name))
            self.SBERT_model.eval()
            self.SBERT_model_dim = self.SBERT_model.encode(
                sentences="test",
                normalize_embeddings=normalization,
                show_progress_bar=False).shape[0]
        if BERT_model_name:
            self.BERT_model = SentenceTransformer(
                str(pretrained_model_path / BERT_model_name))
            self.BERT_model.eval()
            self.BERT_model_dim = self.BERT_model.encode(
                sentences="test",
                normalize_embeddings=normalization,
                show_progress_bar=False).shape[0]
        self.normalization = normalization

    def encode_SBERT(self, sentences: list = [], reduction_dim=-1):
        if isinstance(sentences, str):
            sentences = [sentences]
        sentence_embeddings = self.SBERT_model.encode(
            sentences=sentences,
            normalize_embeddings=self.normalization,
            show_progress_bar=False)
        if reduction_dim > 0 and reduction_dim < self.SBERT_model_dim:
            if len(sentences) == 1:
                sentence_embeddings = sentence_embeddings[0:reduction_dim]
            else:
                sentence_embeddings = self.LSA_dim_reduction(
                    vectors=sentence_embeddings, dim=reduction_dim)
        return sentence_embeddings  # np.array shape = [number of sentences, dim of vector]

    def encode_BERT(self, words: list = [], reduction_dim=-1):
        if isinstance(words, str):
            words = [words]
        word_embeddings = self.BERT_model.encode(
            sentences=words,
            normalize_embeddings=self.normalization,
            show_progress_bar=False)
        if reduction_dim > 0 and reduction_dim < self.SBERT_model_dim:
            if len(words) == 1:
                word_embeddings = word_embeddings[0:reduction_dim]
            else:
                word_embeddings = self.LSA_dim_reduction(
                    vectors=word_embeddings, dim=reduction_dim)
        return word_embeddings  # np.array shape = [number of sentences, dim of vector]

    def LSA_dim_reduction(self, vectors, dim=100):
        if len(vectors.shape) == 1:
            vectors = vectors[np.newaxis, :]
        # svd=TruncatedSVD(n_components=100,random_state=2023)
        svd = make_pipeline(TruncatedSVD(n_components=dim),
                            Normalizer(copy=False))
        embedding = svd.fit_transform(vectors)
        # embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding


pretrained_model_path = Path.cwd() / Configure.get("Embedding", "embedding_models")
if not pretrained_model_path.exists():
    print(
        f"The pretrained Encoder model path {str(pretrained_model_path)} does not exist"
    )
    exit(0)
try:
    BERT_model_name = Configure.get("Embedding", "bert_model")
except Exception as e:
    print(" BERT model name not specified in configuration file")
    BERT_model_name = ''
try:
    SBERT_model_name = Configure.get("Embedding", "sbert_model")
except Exception as e:
    print(" SBERT model name not specified in configuration file")
    SBERT_model_name = ''

# SENTENCE_VECTOR_DIM = int(Configure.get('Embedding', 'sentence_vector_dim'))
# WORD_VECTOR_DIM = int(Configure.get('Embedding', 'word_vector_dim'))
# ACTION_VECTOR_DIM = int(Configure.get('Embedding', 'action_vector_dim'))
encoder = Encoder(pretrained_model_path=pretrained_model_path,
                  BERT_model_name=BERT_model_name,
                  SBERT_model_name=SBERT_model_name)

if __name__ == '__main__':
    text = "21,25,80,139,445"

    vector = encoder.encode_SBERT(sentences=[text]).flatten()
    vector2 = encoder.encode_BERT(words=[text], reduction_dim=50)
    print(vector)
    print(vector2)
