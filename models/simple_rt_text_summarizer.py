import argparse
import pandas as pd
import requests
import re
import time
import sys 
import os
from transformers import pipeline
sys.path.extend(
    [(os.path.abspath(os.path.dirname('__file__')) + "/..")]
)
from data.web_scrapper import ScrapeRt as rt

# Adding arguments to customize CLI 
argparser = argparse.ArgumentParser(description='Process hyper-parameters')
argparser.add_argument('--movie_title', type=str, default='', help='movie title')
argparser.add_argument('--scraping_limit', type=int, default=20, help='scraping limit')
argparser.add_argument('--reviewer', type=str, default='user', help='reviwer type')
argparser.add_argument('--char_limit', type=int, default=20000, help='char limit summary input')
argparser.add_argument('--max_length', type=int, default=150, help='char limit summary output')
args = argparser.parse_args()

print('Scraping Details: ')
print(f'Movie title: {args.movie_title}')
print(f'Number of total reviews attempted to scrape: {args.scraping_limit}')
print(f'Reviews from: {args.reviewer}')
print(f'Character limit for summary text: {args.char_limit}')
print('\n ---------------------')
# print(args.max_length)


#Initializing web scrapper 
scrapper = rt(
    movie_title = args.movie_title ,
    scraping_limit= args.scraping_limit, 
    reviewer = args.reviewer
)

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
    input_text, 
    char_limit = int(), # tested in notebook for tf-5 CUDA mem max limit
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
    print(f'Input text len: {len(processed_text)} \n ')
    summary_text = summarizer(
        processed_text,
        max_length=args.max_length,
        min_length=min_length,
    )[0]['summary_text']

    return summary_text

def execute():
    '''
    Executes main function  by running scrapper.run_for_reviews() & 
    then summarize_text()
    
    Returns 
        dataframe of scrapped data and summary text
    '''
    df = scrapper.run_for_reviews()
    print(scrapper.movie_title)
    reviews = '. '.join([i for i in df.review.values])
    summary = summarize_text(
        summarizer, 
        reviews,
        char_limit= args.char_limit                  
    )
    print('\n --------------------- \n ')
    print(summary)

    return df, summary

if __name__ == "__main__":
    execute()
