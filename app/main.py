# from webbrowser import Konqueror # whats this for ? 
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import numpy as np
import sys 
import os
import textwrap
sys.path.extend(
    [(os.path.abspath(os.path.dirname('__file__')) + "/..")]
)
#custom imports 
from ingestion.web_scrapper import ScrapeRt as rt
from ingestion.web_scrapper import google_movie
from inference.T5FineTunedModel import T5FineTuner
import inference.DimReduction as DimReduction
import inference.TextEmbeddings as TextEmbeddings
import inference.Summarizer as Summarizer 

app = FastAPI()

class Data(BaseModel):
    '''takes in input from API'''
    movie_title: str
    scraping_limit: int
    reviewer: str
    char_limit: int
    max_length: int

@app.get("/")
@app.get("/home")
def read_home():
    return {'message': 'system running successfully'}

@app.post('/predict')
def main(data: Data):
    '''
    Executes main function by running: 
    1. googled_movie_url
    2. scrapper.run_for_reviews() 
    3. TextEmbeddings.get_bert_sentence_embeddings()
    4. DimReduction.run_umap()
    5. DimReduction.get_optimal_gmm()
    6. Summarizer.custom_summarize()
    
    Returns 
        dictionary summaries and relevant data (num reviews, rating, summary)
    '''
    googled_movie_url = google_movie(data.movie_title)
    scrapper = rt(
        movie_title = googled_movie_url,
        scraping_limit= data.scraping_limit, 
        reviewer = data.reviewer
    )
    df = scrapper.run_for_reviews()
    print('Now searching for topics in reviews...')
    sentence_embeddings = TextEmbeddings.get_bert_sentence_embeddings(df['review'])
    umap_embeddings = DimReduction.run_umap(data=sentence_embeddings)
    best_num, best_coeff = DimReduction.get_optimal_gmm(df=umap_embeddings)
    labels = DimReduction.run_gmm(num_clusters=best_num, data=umap_embeddings)
    df['cluster_label'] = labels
    print(f'Number of topics found in reviews: {best_num}')
    loop_order = df['cluster_label'].value_counts().index.to_list() # to start with most predominant topics
    summary_dict = {}
    for i in loop_order:
        df_sample = df[df['cluster_label'] == i].copy()
        reviews = '. '.join([i for i in df_sample.review.values])
        reviews = textwrap.wrap(reviews, data.char_limit)[0] # gpu taps out around this length string 
        summary = Summarizer.custom_summarize(reviews)
        len_stats = np.round(len(df_sample)/len(df)*100, 2)
        rating = np.round(df_sample['rating'].mean(), 2) 
        summary_dict[f'summary{i}'] = {'num_reviews': len_stats, 'rating': rating, 'summary': summary}
    return summary_dict 

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload = True)