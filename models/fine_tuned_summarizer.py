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
from transformers import T5Tokenizer
from T5FineTunedModel import T5FineTuner

# Adding arguments to customize CLI 
argparser = argparse.ArgumentParser(description='Process hyper-parameters')
argparser.add_argument('--movie_title', type=str, default='', help='movie title')
argparser.add_argument('--scraping_limit', type=int, default=30, help='scraping limit')
argparser.add_argument('--reviewer', type=str, default='user', help='reviwer type')
argparser.add_argument('--char_limit', type=int, default=2500, help='char limit summary input')
argparser.add_argument('--max_length', type=int, default=200, help='char limit summary output')
argparser.add_argument('--num_reviews', type=int, default=3, help='number of summaries requested')
args = argparser.parse_args()

print('\n ---------------------')
print('Scraping Details: ')
print(f'Movie title: {args.movie_title}')
print(f'Number of total reviews attempted to scrape: {args.scraping_limit}')
print(f'Reviews from: {args.reviewer}')
print(f'Character limit for summary text: {args.char_limit}')
print(f'Number of summaries to be told: {args.num_reviews}')

#Initializing web scrapper 
# Initializing text summarizer
MODEL_NAME = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
T5_model = T5FineTuner()
# should come from a config file 
trained_model = T5_model.load_from_checkpoint('lightning_logs\\fine_tuning_text_summarizer_xsumdata_v_0_1\\version_1\\checkpoints\\epoch=4-step=4999.ckpt'
)

class Inference():
    def summarize(trained_model, tokenizer, text): 
        text_encoding = tokenizer(
            text,
            max_length = args.max_length, 
            padding = 'max_length', 
            truncation = True, 
            return_attention_mask = True, 
            return_tensors = 'pt'
        )
    #     generated_ids = trained_model.model.generate(
        generated_ids = trained_model.model.generate(
            input_ids=text_encoding['input_ids'], 
            attention_mask = text_encoding['attention_mask'], 
            max_length = args.max_length,
            num_beams = 2,
            repetition_penalty = 2.5,
            length_penalty = 1.0,
    #         early_stopping = True
        )
        preds = [
            tokenizer.decode(gen_id, 
                skip_special_tokens = True, 
                clean_up_tokenization_spaces = True)
            for gen_id in generated_ids
        ]
        return "".join(preds)

    
def generate_sample_dfs(dataframe , num_reviews = 3):
    '''
    Generates n random samples of review dataframe for multiple more 
    appropriately lengthed dataframes
    
    Args: 
        datafrane: pd.df
            input df to be split into multiple randomly sampled dfs  
        num_reviews: int 
            text to be summarized
    Returns
        dictionary with df number as key and dataframe as value 
    
    '''
    num_random_reviews = len(dataframe)//num_reviews
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
    dict_data = generate_sample_dfs(df, args.num_reviews)
    summaries = []
    for i in range(1, args.num_reviews+1):
        df_sample = dict_data[i]
        reviews = '. '.join([i for i in df_sample.review.values])
        reviews = textwrap.wrap(reviews, args.char_limit)[0] # gpu taps out around this length string 
        summary = Inference.summarize(
            trained_model,
            tokenizer, 
            reviews,
#             char_limit= args.char_limit                  
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
