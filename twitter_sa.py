# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:06:25 2017

@author: Aniket
"""

import tweepy
from textblob import TextBlob

# Step 1 - Authenticate
consumer_key= 'lI7sxfC9Jky8l4s92ynxi1WF6'
consumer_secret= 'c1BPCYFEMgZfrtjz7Hr6rcYRc77zOS5OIOqX7a7LlFKRWmmyGx'

access_token='4567521216-nYWaj4NcxpWvHjHWKod8c31fzlOdPUfOzVNNt81'
access_token_secret='T842HsFhYARg0BF3kHjdLwRwtTFaqjUKb5dTtrpisJS60'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('amex')



#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself


for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    if analysis.sentiment.polarity != 0:
        print(analysis.sentiment)
        print("")
        