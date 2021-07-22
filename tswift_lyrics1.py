import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas_bokeh
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


lyrics_df = pd.read_csv('taylor_swift_lyrics.csv',
                        encoding='Latin1')


# typecasting
for col in ['artist', 'album', 'track_title']:
    lyrics_df[col] = lyrics_df[col].astype('category')
    
    
# create album catalog
full_catalog = {}

for album in lyrics_df['album'].unique():
    full_catalog[album] = []
    for track_title in lyrics_df[lyrics_df['album'] == album]['track_title'].unique():
        full_catalog[album].append(track_title)
        
        
# define function to get song lyrics
def get_song_lyrics(track_title, df=lyrics_df):
    full_song = ' '.join(line for line in df[df['track_title'] == track_title]['lyric'])
    return full_song
  
  
# define function to create wordcloud for selected song
def create_song_wordcloud(track_title):
    
    song_lyrics = get_song_lyrics(track_title)
    
    # create mask array of heart shape
    mask_path = Path('heart_shape.png')
    heart_mask = np.array(Image.open(mask_path))
    heart_color = ImageColorGenerator(heart_mask.copy())
    
    # draw wordcloud
    wc = WordCloud(max_words=1000,
               background_color='black',
               mask=heart_mask,
               max_font_size=100,
               relative_scaling=0)
    wc.generate(song_lyrics)
    wc.recolor(color_func=heart_color)
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.imshow(wc, interpolation="bilinear")
    
# define function to get pos tags for selected song
def get_pos_tags(track_title):
    
    song_lyrics = get_song_lyrics(track_title)
    tokens = nltk.word_tokenize(song_lyrics)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags
  
  
# find lemmas and stems of tokens
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


def find_stem_lyrics(track_title):
    
    song_lyrics = get_song_lyrics(track_title)
    tokens = nltk.word_tokenize(song_lyrics)
    
    # initialize stemmer
    stemmer = PorterStemmer() 
    stemmed_tokens = list(set([stemmer.stem(token) for token in tokens]))
    return stemmed_tokens


def find_lemmas_lyrics(track_title):
    
    song_lyrics = get_song_lyrics(track_title)
    tokens = nltk.word_tokenize(song_lyrics)
    
    # initialize Lemmatizer
    lem = WordNetLemmatizer()
    lemmas_tokens = list(set([lem.lemmatize(token) for token in tokens]))
    return lemmas_tokens
  
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def plot_freqdist(track_title, n):
    
    # get tokens from song lyrics
    song_lyrics = get_song_lyrics(track_title)
    
    # initialize tokenizer
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(song_lyrics)
    
    # remove stopwords
    stopwords_list_en = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords_list_en]
    
    # get frequency distribution of tokens
    fd = nltk.FreqDist(tokens)
    return fd.plot(n)
  
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment(text):
    
    # initialize tokenizer
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(text)
    
    sd = SentimentIntensityAnalyzer()
    return sd.polarity_scores(text)
  

lyrics_df['sentiment_score'] = lyrics_df['lyric'].apply(get_sentiment)

def decompose_sentiment_scores(df):
    for sentiment in ['neg', 'neu', 'pos',
                'compound']:
        df[sentiment] = df['sentiment_score'].apply(
                                lambda x: x[sentiment]*100)
    df.rename(columns={'neg': 'negative (%)',
                       'pos': 'positive (%)',
                       'neu': 'neutral (%)',
                       'compound': 'mixed (%)'}, inplace=True)
    

def plot_sentiment_by_track(track_title):
    
    grouped = lyrics_df.groupby('track_title')[['negative (%)', 'neutral (%)', 'positive (%)',
                            'mixed (%)']].mean().reset_index()
    
    grouped[grouped['track_title'] == track_title].plot_bokeh.bar(
            xlabel='Sentiment Score', xticks=[0, 1, 2, 3],
                        
            title='Sentiment score for track: '+ str(track_title))
    
def plot_sentiment_analysis_by_album(album_name='1989'):

  neg_sentiments = grouped[grouped['track_title'].isin(full_catalog[album_name])]['negative (%)']
  pos_sentiments = grouped[grouped['track_title'].isin(full_catalog[album_name])]['positive (%)']

  fig, ax = plt.subplots(figsize=(15, 10))
  ax.barh(np.arange(len(neg_sentiments)),
          -neg_sentiments,
          facecolor='red', alpha=0.7,
          edgecolor='black',
          label='negative')
  ax.barh(np.arange(len(pos_sentiments)),
          pos_sentiments,
          facecolor='green', alpha=0.7,
          edgecolor='black',
          label='positive')

  ax.set_yticks(np.arange(len(pos_sentiments)))
  ax.set_yticklabels(full_catalog['1989'], 
                     fontsize=12)
  ax.set_ylabel('Track Title', fontsize=18)
  ax.set_xlabel('Negative - Positive Sentiment (%)',
                fontsize=18)

  ax.set_title('Average Sentiment Scores for Taylor Swift Album: 1989',
               fontsize=25)
  ax.legend(loc='best')
  plt.show()
