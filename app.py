import sys
#sys.path.append("twint/")
import twint
import textblob
import re
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Set up TWINT config
c = twint.Config()


st.header('Sentiment Analysis App')

st.sidebar.header('Enter Search Term')



Host_Country = st.selectbox('Select Country:', ('Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Ethiopia', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Haiti', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Northern Cyprus', 'Norway', 'Pakistan', 'Palestinian Territories', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Serbia', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Trinidad & Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe')
                            )
st.write('You selected:', Host_Country)


key_words   = [" food shortage", " food insecurity", " malnutrition", " hunger"," famine"]

def search_job():
      for i in key_words:
            word_search = Host_Country+i

            search_term = str(word_search)
            c.Search = str(search_term)
            c.Limit = 100
            c.Pandas = True

            twint.run.Search(c)

search_job()

def available_columns():
      return twint.output.panda.Tweets_df.columns

def twint_to_pandas(columns):
      return twint.output.panda.Tweets_df[columns]

news_data = pd.read_csv("labeled_food_news2.csv")

vectorizer = TfidfVectorizer()

df_pd = twint_to_pandas(["date", "username", "tweet", "hashtags", "nlikes"])

def clean_tweet(tweet):
      return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_sentiment(tweet):

      # create TextBlob object of passed tweet text
      analysis = textblob.TextBlob(clean_tweet(tweet))
       # set sentiment
      if analysis.sentiment.polarity > 0:
            return 'positive'
      elif analysis.sentiment.polarity == 0:
            return 'neutral'
      else:
            return 'negative'


df_pd['sentiment'] = df_pd.tweet.apply(lambda twt: get_sentiment(twt))

#write percentages of tweets according to sentiments
positive = (len(df_pd[df_pd['sentiment'] == 'positive']) /
            len(df_pd['sentiment'])) * 100
st.sidebar.write(round(positive, 2), '% Positive')

neutral = (len(df_pd[df_pd['sentiment'] == 'neutral']) /
           len(df_pd['sentiment'])) * 100
st.sidebar.write(round(neutral, 2), '% Neutral')

negative = (len(df_pd[df_pd['sentiment'] == 'negative']) /
            len(df_pd['sentiment'])) * 100
st.sidebar.write(round(negative, 2), '% Negative')

st.subheader('Pie Chart')
fig, ax = plt.subplots()
ax.pie([positive, negative, neutral], radius=0.7, autopct='%1.2f%%')
st.pyplot(fig)

st.subheader('Wordcloud')
cloud_tweets = df_pd.tweet.apply(lambda twt: clean_tweet(twt))
wordcloud = WordCloud(background_color='white', width= 800, height = 350,).generate(str(cloud_tweets))
st.image(wordcloud.to_array())

P_t = df_pd[df_pd['nlikes'] == df_pd['nlikes'].max()]
news_vectors = vectorizer.fit_transform(news_data['processed_summary'].astype('U').values)
tweet_vector = vectorizer.transform(P_t['tweet'])

similarities = cosine_similarity(tweet_vector, news_vectors)

str_sim = [str(i) for i in similarities]
closest = np.argmax(similarities, axis=1)

st.write(str(str_sim[:10]))
st.subheader(str(df_pd[df_pd['nlikes'] == df_pd['nlikes'].max()].astype('U').values))

st.subheader("Article 1:", str(news_data['title'].iloc[closest].values[0]))

st.write(str(news_data['summary'].iloc[closest].values[0]))

st.write(str(news_data['link'].iloc[closest].values[0]))

st.write(str(news_data['published_date'].iloc[closest].values[0]))

st.subheader("Article 2:", str(news_data['title'].iloc[closest].values[0][1]))

st.write(str(news_data['summary'].iloc[closest].values[0][1]))

st.write(str(news_data['link'].iloc[closest].values[0][1]))

st.write(str(news_data['published_date'].iloc[closest].values[0][1]))
'''
st.subheader("Article 3:", str(news_data['title'].iloc[closest].values[2]))

st.write(str(news_data['summary'].iloc[closest].values[2]))

st.write(str(news_data['link'].iloc[closest].values[2]))

st.write(str(news_data['published_date'].iloc[closest].values[2]))
'''
st.write(len(df_pd))



