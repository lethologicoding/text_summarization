# streamlit app
import streamlit as st
import requests 
import json 
import pandas as pd

# Fast API endpoint
APP_URL = "http://127.0.0.1:8000/predict"

#for hiding df index in UI
hide_table_row_index = """
    <style>
    tbody th {display:none}
    .blank {display:none}
    </style>
    """

def run():
    st.title("Movie Review Summarizer")
    st.write("")
    st.write("")
    st.write("""Please enter a **movie title**, 
            kind of **reviewer type** you'd like to gist of, 
            and **how many** recent reviews to examine. 
            Once all info is entered, press **"Predict"** . """ )
    st.write("")
    st.write("")
    st.write("""While the tool is running, it is scraping rotten tomatoes for movie reviews, 
                automatically finding the number of topics/clusters to those reviews, 
                then summarizing the reviews by the found topics/clusters.""") 
    st.write("")
    st.write("")
    movie_title = st.text_input("Enter Movie")
    reviewer = st.selectbox("Enter Reviewer" , ["Audience"] ) # , "Critics"])
    scraping_limit = st.slider('Select amount of recent reviews to examine', 10, 100, step = 10)

    if reviewer == 'Audience':
        reviewer = 'user'

    Data = {
    "movie_title": movie_title, 
    "reviewer": reviewer,
    "scraping_limit": scraping_limit,
    "char_limit": 100,
    "max_length": 100
    }

    if st.button("Predict"):
        print(Data)
        response=requests.post(APP_URL, json=Data)
        decoded_output=response.content.decode('UTF-8')
        output=json.loads(decoded_output)
        df = pd.DataFrame(output).T.round(2)
        df.columns = ["% of reviews", "Avg. Rating", 'Summary']
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(df.style.format({"% of reviews": '{:.0f}',
                                     "Avg. Rating": '{:.2f}'
                }))


if __name__ == '__main__':
    run()