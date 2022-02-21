import requests
import argparse
import json
import pandas as pd

import streamlit as st


# Adding arguments to customize CLI 
argparser = argparse.ArgumentParser(description='Process hyper-parameters')
argparser.add_argument('--movie_title', type=str, default='', help='movie title')
argparser.add_argument('--scraping_limit', type=int, default=10, help='scraping limit')
argparser.add_argument('--reviewer', type=str, default='user', help='reviwer type')
argparser.add_argument('--char_limit', type=int, default=30000, help='char limit summary input')
argparser.add_argument('--max_length', type=int, default=100, help='char limit summary output')
args = argparser.parse_args()

print('\n ---------------------')
print('Scraping Details: ')
print(f'Movie title: {args.movie_title}')
print(f'Number of total reviews attempted to scrape: {args.scraping_limit}')
print(f'Reviews from: {args.reviewer}')
print(f'Character limit for summary text: {args.char_limit}')


payload = {
    'movie_title': args.movie_title,
    'scraping_limit': args.scraping_limit,
    'reviewer': args.reviewer,
    'char_limit': args.char_limit, 
    'max_length':args.max_length
}


response=requests.post('http://localhost:7000/predict', json=payload)
decoded_output=response.content.decode('UTF-8')
output=json.loads(decoded_output)
print(output)