import argparse
from asyncore import loop
import numpy as np
import sys 
import os
import textwrap
from transformers import T5Tokenizer
sys.path.extend(
    [(os.path.abspath(os.path.dirname('__file__')) + "/..")]
)
#custom imports 
from ingestion.web_scrapper import ScrapeRt as rt
from ingestion.web_scrapper import google_movie
from T5FineTunedModel import T5FineTuner
import DimReduction
import TextEmbeddings

# Adding arguments to customize CLI 
argparser = argparse.ArgumentParser(description='Process hyper-parameters')
argparser.add_argument('--movie_title', type=str, default='', help='movie title')
argparser.add_argument('--scraping_limit', type=int, default=30, help='scraping limit')
argparser.add_argument('--reviewer', type=str, default='user', help='reviwer type')
argparser.add_argument('--char_limit', type=int, default=30000, help='char limit summary input')
argparser.add_argument('--max_length', type=int, default=100, help='char limit summary output')
# argparser.add_argument('--num_reviews', type=int, default=3, help='number of summaries requested')
args = argparser.parse_args()

print('\n ---------------------')
print('Scraping Details: ')
print(f'Movie title: {args.movie_title}')
print(f'Number of total reviews attempted to scrape: {args.scraping_limit}')
print(f'Reviews from: {args.reviewer}')
print(f'Character limit for summary text: {args.char_limit}')
# print(f'Number of summaries to be told: {args.num_reviews}')

#Initializing web scrapper 
# Initializing text summarizer
#NOTE: should come from a config file 
MODEL_NAME = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
T5_model = T5FineTuner()
trained_model = T5_model.load_from_checkpoint('..\\training\\lightning_logs\\fine_tuning_text_summarizer_rt_v_0_2\\version_0\\checkpoints\\epoch=19-step=4319.ckpt')

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
        generated_ids = trained_model.model.generate(
            input_ids=text_encoding['input_ids'], 
            attention_mask = text_encoding['attention_mask'], 
            max_length = args.max_length,
            num_beams = 2,
            repetition_penalty = 2.5,
            length_penalty = 1.0,
        )
        preds = [
            tokenizer.decode(gen_id, 
                skip_special_tokens = True, 
                clean_up_tokenization_spaces = True)
            for gen_id in generated_ids
        ]
        return "".join(preds)

def execute():
    '''
    Executes main function by running scrapper.run_for_reviews() & 
    then summarize_text()
    
    Returns 
        dataframe of scrapped data and summary text
    '''
    print('\n --------------------- \n ')
    print('\n **Started Reviewing**') 
    print('Googling movie for rotten tomatoes url')
    googled_movie_url = google_movie(args.movie_title)
    print(f'Search found for {googled_movie_url}')
    scrapper = rt(
        movie_title = googled_movie_url,
        scraping_limit= args.scraping_limit, 
        reviewer = args.reviewer
    )
    df = scrapper.run_for_reviews()
    print('Now searching for topics in reviews...')
    sentence_embeddings = TextEmbeddings.get_bert_sentence_embeddings(df['review'])
    ## fitting high dimensional sentence embeddings into umap'd 2D space
    umap_embeddings = DimReduction.run_umap(data=sentence_embeddings)
    best_num, best_coeff = DimReduction.get_optimal_gmm(df=umap_embeddings)
    labels = DimReduction.run_gmm(num_clusters=best_num, data=umap_embeddings)
    df['cluster_label'] = labels
    print(f'Number of topics found in reviews: {best_num}')
    loop_order = df['cluster_label'].value_counts().index.to_list() # to start with most predominant topics 

    summaries = []
    for i in loop_order:
        df_sample = df[df['cluster_label'] == i].copy()
        reviews = '. '.join([i for i in df_sample.review.values])
        reviews = textwrap.wrap(reviews, args.char_limit)[0] # gpu taps out around this length string 
        summary = Inference.summarize(
            trained_model,
            tokenizer, 
            reviews,
        )
        print('\n --------------------- \n ')
        len_stats = np.round(len(df_sample)/len(df)*100, 2)
        rating_stats = np.round(df_sample['rating'].mean(), 2) 
        print(f'Topic had {len_stats}% reviews with an avg. rating of: {rating_stats}. \n')
        print('Summary of topic-based reviews: \n ')
        print(summary)
        summaries.append(summary)
    print('\n --------------------- \n ')
    print('\n **Completed Reviewing**') 
    print('\n --------------------- \n ')
    return summaries

if __name__ == "__main__":
    execute()
