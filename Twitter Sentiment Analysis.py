# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Convolution2D,Dropout,MaxPooling2D,Flatten,Convolution1D,MaxPooling1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

#Fetch Dataset


tweet_data = pd.read_csv('train(1).csv',encoding='latin-1')
tweet_arr = np.array(tweet_data.as_matrix())
np.random.shuffle(tweet_arr)
tweet_data = pd.DataFrame(tweet_arr,columns=tweet_data.columns)
del tweet_arr
#Data Preprocessing

tweet_data=tweet_data[tweet_data['SentimentText'].apply(lambda x: len(x)<480)]
tweets = tweet_data['SentimentText'].iloc[:60000].values
tweets = tweets.tolist()
X = []

for item in tweets:
    X.append(item.lower())

tk = Tokenizer(char_level=True,filters=''' $%&()*+,-./:;<=>?[\]^_`{|}~ ''',lower=True)
tk.fit_on_texts(X)

encoding_len = len(tk.word_counts)+1

data = np.zeros((len(X),480,encoding_len,1))

for index,item in enumerate(X):
    temp = tk.texts_to_matrix(item)
    row,_ = temp.shape
    data[index]= np.pad(temp,((0,(480-row)),(0,0)),mode='constant')

y = tweet_data['Sentiment'][:60000]

model = Sequential()
model.add(Convolution1D(filters=64,kernel_size=7,activation='relu',input_shape=(480,encoding_len)))
model.add(MaxPooling1D(pool_size=3))
model.add(Convolution1D(filters=64,kernel_size=7,activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Convolution1D(filters=32,kernel_size=3,activation='relu'))
model.add(Convolution1D(filters=32,kernel_size=3,activation='relu'))
model.add(Convolution1D(filters=16,kernel_size=3,activation='relu'))
model.add(Convolution1D(filters=16,kernel_size=3,activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(activation='relu',units=64))
model.add(Dropout(0.5))
model.add(Dense(activation='relu',units=32))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

with tf.device('/gpu:0'):
    model.fit(data,y,epochs=10,validation_split=0.1,batch_size=64)

pred_item = tweet_data.iloc[50002]['SentimentText']
item = tk.texts_to_matrix(pred_item)
row,_ = item.shape
item = np.pad(item,((0,(480-row)),(0,0)),mode='constant')
item = np.reshape(item,(480,encoding_len,1))
item = np.expand_dims(a=item,axis=0)

print(model.predict(item)>0.5)
