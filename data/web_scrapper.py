import pandas as pd
import requests
import re
import time
session = requests.Session() 

class ScrapeRt(): 
    '''
    Instance scrapes rotten tomatoes reviews
    ... 
    Attributes: 
        movie_title: str 
            Title of the movie, 
        reviewer: str 
            Select reviewer pool from list: ['critic', 'user']. Default is user. 
        scrape_limit: int 
            Number of pages to stop scraping. Default is 10. 
        write_data : boolean
            Writes data out to pickle object. Default is False. 
    Methods: 
        scrape_reviews
            Scrapes rotten tomatoe page
        filter_df
            Filters scraped df 
        write_df
            Writes out data
        run_for_reviews
            Runs all methods above
    '''
    def __init__(
        self, 
        movie_title = '',
        reviewer = 'user',
        scraping_limit = 10, 
        write_data = False
    ):
    
        self.movie_title = movie_title
        self.reviewer = reviewer
        self.scraping_limit = scraping_limit
        self.write_data = write_data
        self.review_df = None
        print('Scrapped Initiated')
        print(f'Scrapping data for: {self.movie_title}')
        
    def scrape_reviews(self):
        '''
        Scrapes rotten tomatoes for reviews and related info, and updates self.review_df with data 
        '''
        url = f'https://www.rottentomatoes.com/m/{self.movie_title}/reviews'
        r = requests.get(url)
        movie_id = re.findall(r'(?<=movieId":")(.*)(?=","type)',r.text)[0]

        if self.reviewer == 'critic':
            api_url = f"https://www.rottentomatoes.com/napi/movie/{movie_id}/criticsReviews/all"
        if self.reviewer == 'user':
            api_url = f"https://www.rottentomatoes.com/napi/movie/{movie_id}/reviews/user"
        payload = {
            'direction': 'next',
            'endCursor': '',
            'startCursor': '',
        }
        pages_scraped = 0
        review_data = []
        while True:
            r = session.get(api_url, 
                      params=payload)
            data = r.json()

            if not data['pageInfo']['hasNextPage']:
                print('Scaping completed')
                break
            elif pages_scraped == self.scraping_limit:
                print('Scraping limit reached')
                break

            payload['endCursor'] = data['pageInfo']['endCursor']
            payload['startCursor'] = data['pageInfo']['startCursor'] if data['pageInfo'].get('startCursor') else ''
            review_data.extend(data['reviews'])
            time.sleep(.1)
            pages_scraped += 1
            if pages_scraped % 10 == 0: 
                print(f'Pages scraped: {pages_scraped}')
#         print(review_data)
        self.review_df =  pd.json_normalize(review_data)
        return 
        
    def filter_df(self):
        '''
        takes in self.review_df and updates it based on filtering conditionals
        
        NOTE: if there are not enough verified user reviews (>5), no filtering carried out. 
        '''
    
        if self.reviewer == 'user':
            filtered_df = self.review_df[self.review_df['isVerified'] == True]
            if filtered_df.shape[0] < 5:
                self.review_df = self.review_df
                print('Not enough verified reviews... Sticking to all reviews')
        return
    
    def write_df(self):
        '''
        If self.write_data == True, a pickled dataframe will be written
        '''
        if self.write_data == True:
            self.review_df.to_pickle(f'{self.movie_title}_data.pkl')
    
    def run_for_reviews(self):
        '''
        runs all class methods -> scrape_reviews, filter_df, and write_df 
        '''
        try: 
            self.scrape_reviews()
            self.filter_df()
            self.write_df()
            return self.review_df
        
        except: 
            print(f'Could not find reviews for *{self.movie_title}*... Please try to find another one' )
        
