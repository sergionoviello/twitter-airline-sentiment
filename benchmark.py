import numpy as np
import pandas as pd
import sys
import json
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
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
BATCH_SIZE= 32
EMBED_DIMS = 200

MODEL_FILE = 'model.hdf5'
CHECKPOINT_FILE = 'checkpoint.hdf5'
VECTORIZER_FILE = 'vect.pkl'

class AirlineSentiment:
  def __init__(self, text_preprocessor, word_to_vec):
    self.df = self.get_data()
    self.df['max_len'] = self.df['text'].apply(lambda x: len(x))
    self.max_len = self.df['max_len'].max()

    sentiment_map = {'negative':0, 'neutral':1, 'positive':2}
    self.df['airline_sentiment'] = self.df['airline_sentiment'].map(sentiment_map)
    self.clean_text(text_preprocessor)
    self.df = self.df[~self.df['clean_text'].apply(self.is_not_ascii)]
    self.df = self.df[pd.notnull(self.df['clean_text'])]

    self.w2v_file = word_to_vec.train(self.df['clean_text'])

 
    
  def train(self):
    w2v_model = word2vec.Word2Vec.load(W2V_MODEL_FILE)
    vocab_size = len(w2v_model.wv.vocab)
    max_len = self.df['max_len'].max()

    X = self.text_to_word_embeddings(self.df['text'].values, vocab_size, self.max_len)
    y = np_utils.to_categorical(self.df['airline_sentiment'].values)
    X_train, X_val, Y_train, Y_val = train_test_split(X,y, test_size = 0.3, random_state = 42)

    #embedding matrix
    
    self.embed_matrix = np.zeros((len(w2v_model.wv.vocab), EMBED_DIMS))
    for i in range(len(w2v_model.wv.vocab)):
        embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
        if embedding_vector is not None:
            self.embed_matrix[i] = embedding_vector

    print('embedding matrix shape:', self.embed_matrix.shape)

    model = self.build_model(max_len)

    filepath="saved_models/{}".format(CHECKPOINT_FILE)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), callbacks = [checkpoint])

    model.save("saved_models/{}".format(MODEL_FILE))

    score,acc = model.evaluate(X_val, Y_val, verbose = 2, batch_size = BATCH_SIZE)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

  def build_model(self, max_len):
    model = Sequential()
    # input_dim' = the vocab size that we will choose. In other words it is the number of unique words in the vocab.
    # 'output_dim' = the number of dimensions we wish to embed into. Each word will be represented by a vector of this much dimensions.
    # An example of shape of embeddings
    # The resulting shape is (3,12,8).
    # 3---> no of documents
    # 12---> each document is made of 12 words which was our maximum length of any document.
    # & 8---> each word is 8 dimensional.
    model.add(Embedding(input_dim=self.embed_matrix.shape[0], output_dim=self.embed_matrix.shape[1], input_length=max_len, weights=[self.embed_matrix], trainable=False))

    #model.add(Dropout(0.5))
    model.add(LSTM(EMBED_DIMS, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))

    # adam default parameters:  lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

  def predict_single_text(self, text):
    model = load_model("saved_models/{}".format(MODEL_FILE))
    with open("saved_models/{}".format(VECTORIZER_FILE), 'rb') as f2:
      vect = pickle.load(f2)
    sequences = vect.texts_to_sequences([text])
    X_test = pad_sequences(sequences, maxlen=self.max_len)
    print('predict...')
    pred = model.predict(X_test)[0]
    prob_map = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    print('****************')
    print(prob_map[np.argmax(pred)])
    print('****************')

  
  def text_to_word_embeddings(self, texts, vocab_size, max_len):
    self.tokenizer = Tokenizer(num_words=vocab_size)
    self.tokenizer.fit_on_texts(texts)

    sequences = self.tokenizer.texts_to_sequences(texts)
    
    x_train = pad_sequences(sequences, maxlen=max_len)

    with open("saved_models/{}".format(VECTORIZER_FILE), 'wb') as handle:
      pickle.dump(self.tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
      print ('tokenizer saved')

    return x_train
  
  def is_not_ascii(self, string):
    return string is not None and any([ord(s) >= 128 for s in string])

  def metrics(self):
    pos_count = self.df[self.df.airline_sentiment == 'positive']['tweet_id'].count()
    neg_count = self.df[self.df.airline_sentiment == 'negative']['tweet_id'].count()
    neut_count = self.df[self.df.airline_sentiment == 'neutral']['tweet_id'].count()

    print("tot: {}, pos: {}, neutr:{}, neg: {}".format(self.df.shape[0], pos_count, neut_count, neg_count))
    print('max sentence length', self.df['max_len'].max())

    #length of tweets

    reviews_len = self.df['max_len'].values #[len(x) for x in reviews_int]
    pd.Series(reviews_len).hist()
    #plt.show()
    print(pd.Series(reviews_len).describe())
    plt.savefig('tweets_len.png')

  def get_data(self):
    return pd.read_csv('data/Tweets.csv') 

  def clean_text(self, text_preprocessor):
    clean_text = text_preprocessor.pre_process(self.df['text'])
    self.df['clean_text'] = clean_text

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


W2V_SIZE = EMBED_DIMS
W2V_WINDOW = 7
W2V_EPOCH = 64
W2V_MIN_COUNT = 3

W2V_FILE = "saved_models/{}".format('airlines.w2v')
W2V_MODEL_FILE = "saved_models/{}".format('airlines_model.w2v')
class Word2VecCreator:
  def __init__(self):
    pass

  def train(self, text_series):
    self.documents = [_text.split() for _text in text_series]
    w2v_model = word2vec.Word2Vec(
                                  size=W2V_SIZE,
                                  #window=W2V_WINDOW,
                                  min_count=W2V_MIN_COUNT,
                                  workers=8)
    w2v_model.build_vocab(self.documents)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size:", vocab_size)
    w2v_model.train(self.documents, total_examples=len(self.documents), epochs=W2V_EPOCH)

    print('Embedding matrix shape:', w2v_model.wv.syn0.shape)

    
    

    w2v_model.wv.save_word2vec_format(W2V_FILE, binary=False)
    w2v_model.save(W2V_MODEL_FILE)
    return W2V_FILE

  def test_model(self):
    w2v_model = word2vec.Word2Vec.load(W2V_MODEL_FILE)
    w1 = ['bad']  
    test = w2v_model.wv.most_similar(positive=w1, topn=6)
    print(test)

    print(w2v_model.wv.similarity(w1='good', w2='good'))

tp = TextPreprocessor()
wv = Word2VecCreator()
a = AirlineSentiment(tp, wv)
# a.metrics()

a.train()
#a.train_kfold()
# a.predict_single_text('I am happy')

# # debug Word2VecCreator
# docs = ['the cat sat on the bench', 'anarchism originated as a term of abuse']
# wv.train(docs)


#####LATEST SCORE
# score: 0.83
# acc: 0.78