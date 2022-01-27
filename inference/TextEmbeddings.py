import pandas as pd
from sentence_transformers import SentenceTransformer


def get_bert_sentence_embeddings(df = pd.Series):
    '''
    Uses bert sentence embeddings to transform raw text into embeddings
    '''
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = sbert_model.encode(df)
    
    return sentence_embeddings
