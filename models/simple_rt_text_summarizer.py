import argparse
import numpy as np
import pandas as pd
import requests
import re
import time
import sys 
import os
from nltk import sent_tokenize
import textwrap
from transformers import pipeline
sys.path.extend(
    [(os.path.abspath(os.path.dirname('__file__')) + "/..")]
)
from data.web_scrapper import ScrapeRt as rt

# Adding arguments to customize CLI 
argparser = argparse.ArgumentParser(description='Process hyper-parameters')
argparser.add_argument('--movie_title', type=str, default='', help='movie title')
argparser.add_argument('--scraping_limit', type=int, default=30, help='scraping limit')
argparser.add_argument('--reviewer', type=str, default='user', help='reviwer type')
argparser.add_argument('--char_limit', type=int, default=1500, help='char limit summary input')
argparser.add_argument('--max_length', type=int, default=200, help='char limit summary output')
argparser.add_argument('--num_sum', type=int, default=3, help='number of summaries requested')
args = argparser.parse_args()

print('\n ---------------------')
print('Scraping Details: ')
print(f'Movie title: {args.movie_title}')
print(f'Number of total reviews attempted to scrape: {args.scraping_limit}')
print(f'Reviews from: {args.reviewer}')
print(f'Character limit for summary text: {args.char_limit}')
print(f'Number of summaries to be told: {args.num_sum}')

#Initializing web scrapper 
# Initializing text summarizer
summarizer = pipeline(
    "summarization", 
    model="t5-base", 
    tokenizer="t5-base", 
    device = 0, 
    clean_up_tokenization_spaces  = True # comment out line if there is no CUDA enabled GPU
)

def summarize_text(
    summarizer,
    input_text = str, 
    char_limit = int, # tested in notebook for tf-5 CUDA mem max limit
    max_length=150,
    min_length=50):
    '''
    Summarizes text using initiated torch.pipeline model
    
    Args: 
        summarizer: model
            text summarization model 
        input_text: str 
            text to be summarized
        char_limit: int
            len of text fed into model 
        max_length: int
            max len of the summary returned
        min_length= int
            min len of the summary returned
    Returns
        Summary text 
    '''
    processed_text = input_text[:char_limit] # gpu taps out around this length string 
#     print(f'Input text len: {len(processed_text)} \n ')
    summary_text = summarizer(
        processed_text,
        max_length=args.max_length,
        min_length=min_length,
    )[0]['summary_text']

    return summary_text

def generate_sample_dfs(dataframe , num_sum = 3):
    '''
    Generates n random samples of review dataframe for multiple more 
    appropriately lengthed dataframes
    
    Args: 
        datafrane: pd.df
            input df to be split into multiple randomly sampled dfs  
        num_sum: int 
            text to be summarized
    Returns
        dictionary with df number as key and dataframe as value 
    
    '''
    num_random_reviews = len(dataframe)//num_sum
    samp_range = len(dataframe) //num_random_reviews
    df_shuff = dataframe.reindex(np.random.permutation(dataframe.index))
 

    df_dict = {}
    for i in range(samp_range): 
        df_dict[i+1] = df_shuff[i*num_random_reviews:(i+1)*num_random_reviews]

    return df_dict

def execute():
    '''
    Executes main function by running scrapper.run_for_reviews() & 
    then summarize_text()
    
    Returns 
        dataframe of scrapped data and summary text
    '''
    print('\n --------------------- \n ')
    print('\n **Started Reviewing**') 
    scrapper = rt(
        movie_title = args.movie_title ,
        scraping_limit= args.scraping_limit, 
        reviewer = args.reviewer
    )
    df = scrapper.run_for_reviews()
    dict_data = generate_sample_dfs(df, args.num_sum)
    summaries = []
    for i in range(1, args.num_sum+1):
        df_sample = dict_data[i]
        reviews = '. '.join([i for i in df_sample.review.values])
        reviews = textwrap.wrap(reviews, args.char_limit)[0] # gpu taps out around this length string 
        summary = summarize_text(
            summarizer, 
            reviews,
            char_limit= args.char_limit                  
        )
        print('\n --------------------- \n ')
        print(summary)
        summaries.append(summary)
    print('\n --------------------- \n ')
    print('\n **Completed Reviewing**') 
    print('\n --------------------- \n ')
    return dict_data, summaries

if __name__ == "__main__":
    execute()
