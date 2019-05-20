import numpy as np
import pandas as pd
import sys
import json
import re
import pickle
import collections

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.models import load_model

from sklearn.model_selection import train_test_split, StratifiedKFold

from gensim.models import word2vec
from nltk.stem.wordnet import WordNetLemmatizer

import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

EPOCHS = 10
BATCH_SIZE= 128
EMBED_DIMS = 200

MODEL_FILE = 'model.hdf5'
CHECKPOINT_FILE = 'checkpoint.hdf5'
VECTORIZER_FILE = 'vect.pkl'

class AirlineSentiment:
  def __init__(self, text_preprocessor):
    self.df = self.get_data()
    self.df['max_len'] = self.df['text'].apply(lambda x: len(x))
    self.max_len = self.df['max_len'].max()

    sentiment_map = {'negative':0, 'neutral':1, 'positive':2}
    self.df['airline_sentiment'] = self.df['airline_sentiment'].map(sentiment_map)
    self.clean_text(text_preprocessor)

    self.df = self.df[pd.notnull(self.df['clean_text'])]

  def train(self):
    num_steps=30 #the set of words that the model will learn from to predict the words coming after
    batch_size=20
    hidden_size=EMBED_DIMS
    data_path = 'saved_models'
    num_epochs = EPOCHS
    use_dropout = False
    train_data, val_data, vocabulary, reversed_dictionary = self.format_data()
    
    train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary, skip_step=num_steps)
    valid_data_generator = KerasBatchGenerator(val_data, num_steps, batch_size, vocabulary, skip_step=num_steps)

    model = Sequential()
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(val_data)//(batch_size*num_steps), callbacks=[checkpointer])
 

  def format_data(self):
    train, val = train_test_split(self.df, test_size=0.2)
    train_list = self.text_to_list(train)
    val_list = self.text_to_list(val)

    word_to_id = self.build_vocab(train_list) # creates a dictionary of words and their index: {'what': 9, 'plus': 3, 'tacky': 5, 've': 8, 'to': 7, 'you': 10,
    
    train_data = self.file_to_word_ids(train_list, word_to_id) #list of indices instead of words
    val_data = self.file_to_word_ids(val_list, word_to_id)

    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, val_data, vocabulary, reversed_dictionary

  def text_to_list(self, df):
    data = df['clean_text']
    data = [_text.split() for _text in data]
    data = [item for sublist in data for item in sublist]
    return data

  def build_vocab(self, data):
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

  def file_to_word_ids(self, data, word_to_id):
    
    return [word_to_id[word] for word in data if word in word_to_id]

  def get_data(self):
    return pd.read_csv('data/Tweets.csv') 

  def clean_text(self, text_preprocessor):
    clean_text = text_preprocessor.pre_process(self.df['text'])
    self.df['clean_text'] = clean_text

class KerasBatchGenerator(object):
  def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
    self.data = data
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.vocabulary = vocabulary
    # this will track the progress of the batches sequentially through the
    # data set - once the data reaches the end of the data set it will reset
    # back to zero
    self.current_idx = 0
    # skip_step is the number of words which will be skipped before the next
    # batch is skimmed from the data set
    self.skip_step = skip_step

  def generate(self):
    x = np.zeros((self.batch_size, self.num_steps))
    y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
    while True:
      for i in range(self.batch_size):
        if self.current_idx + self.num_steps >= len(self.data):
            # reset the index back to the start of the data set
            self.current_idx = 0
        x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
        temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
        # convert all of temp_y into a one hot representation
        y[i, :, :] = np_utils.to_categorical(temp_y, num_classes=self.vocabulary)
        self.current_idx += self.skip_step
      yield x, y


class TextPreprocessor:
  def __init__(self):
    with open('abbreviation.json', 'r') as f:
      self.abbr = json.load(f)

  def pre_process(self, data):
    return data.apply(self.pre_process_text)

  def pre_process_text(self, text):
    stops = set(stopwords.words("english"))
    text = text.lower() # lower case
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) # remove links urls

    #remove punctuation
    tokens = nltk.word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    #remove stop words
    words = [w for w in words if not w in stops]

    #lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(t) for t in words]

    words = text.split()
    return ' '.join(words)


tp = TextPreprocessor()
a = AirlineSentiment(tp)
a.train()
# a.metrics()

#a.train()
#a.train_kfold()
# a.predict_single_text('I am happy')

# # debug Word2VecCreator
# docs = ['the cat sat on the bench', 'anarchism originated as a term of abuse']
# wv.train(docs)
