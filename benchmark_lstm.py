import numpy as np
import pandas as pd
import sys
import json
import re
import pickle
import time
import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.models import load_model

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight as cw

from gensim.models import word2vec
from nltk.stem.wordnet import WordNetLemmatizer

import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

EPOCHS = 10
BATCH_SIZE= 512
EMBED_DIMS = 200

MAX_LEN = 186

MODEL_FILE = 'model.hdf5'
CHECKPOINT_FILE = 'checkpoint.hdf5'
VECTORIZER_FILE = 'vect.pkl'

class AirlineSentiment:
  def __init__(self, text_preprocessor):
    self.df = self.get_data()
    self.df['max_len'] = self.df['text'].apply(lambda x: len(x))

    sentiment_map = {'negative':0, 'neutral':1, 'positive':2}
    self.df['airline_sentiment'] = self.df['airline_sentiment'].map(sentiment_map)
    self.clean_text(text_preprocessor)
    self.df = self.df[~self.df['clean_text'].apply(self.is_not_ascii)]
    self.df = self.df[pd.notnull(self.df['clean_text'])]

  def train(self):
    embed_dict = self.create_word_embeddings_dict()
    vocab_size = len(embed_dict.keys())
    print('VOCAB SIZE:', vocab_size)
    max_len = MAX_LEN

    X = self.text_to_word_embeddings(self.df['text'].values, vocab_size)
    y = np_utils.to_categorical(self.df['airline_sentiment'].values)
    X_train, X_val, Y_train, Y_val = train_test_split(X,y, test_size = 0.3, random_state = 42)

    #embedding matrix
    self.embed_matrix = np.zeros((vocab_size, EMBED_DIMS))
    for w, i in self.tokenizer.word_index.items():
        if i < vocab_size:
            vect = embed_dict.get(w)
            if vect is not None:
              self.embed_matrix[i] = vect
        else:
            break

    print('embedding matrix shape:', self.embed_matrix.shape)

    model = self.build_model()


    filepath="saved_models/{}".format(CHECKPOINT_FILE)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    class_weight = self.get_weight(Y_train.flatten())

    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), callbacks = [checkpoint], class_weight=class_weight)

    model.save("saved_models/{}".format(MODEL_FILE))

    # score,acc = model.evaluate(X_val, Y_val, verbose = 2, batch_size = BATCH_SIZE)
    # print("score: %.2f" % (score))
    # print("acc: %.2f" % (acc))

    #self.plot_performance(history, 'saved_models')

    model2 = self.get_rnn_model()
    history2 = model2.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), callbacks = [checkpoint], class_weight=class_weight)
    self.compare_models(history, history2)

  def create_word_embeddings_dict(self):
    filename = "data/{}".format('glove.twitter.27B.200d.txt')
    emb_dict = {}
    glove = open(filename, 'r', encoding = "utf-8")
    for line in glove:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        emb_dict[word] = vector
    glove.close()
    return emb_dict

  def build_model(self):
    model = Sequential()
    # input_dim' = the vocab size that we will choose. In other words it is the number of unique words in the vocab.
    # 'output_dim' = the number of dimensions we wish to embed into. Each word will be represented by a vector of this much dimensions.
    # An example of shape of embeddings
    # The resulting shape is (3,12,8).
    # 3---> no of documents
    # 12---> each document is made of 12 words which was our maximum length of any document.
    # & 8---> each word is 8 dimensional.
    model.add(Embedding(input_dim=self.embed_matrix.shape[0], output_dim=self.embed_matrix.shape[1], input_length=MAX_LEN, weights=[self.embed_matrix], trainable=False))

    #model.add(LSTM(EMBED_DIMS, dropout=0.2, recurrent_dropout=0.2))
    #model.add(Dense(3,activation='softmax'))

    model.add(LSTM(EMBED_DIMS, return_sequences=True))
    model.add(LSTM(EMBED_DIMS, return_sequences=False))
    # if use_dropout:
    #model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))

    # adam default parameters:  lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0................................
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

  def get_rnn_model(self):
    model = Sequential()

    model.add(Embedding(input_dim=self.embed_matrix.shape[0], output_dim=self.embed_matrix.shape[1], input_length=MAX_LEN, weights=[self.embed_matrix], trainable=False))
    model.add(LSTM(EMBED_DIMS))

    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(3,activation='softmax'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.summary()

    return model

  def build_regularization_model(self):
    model = Sequential()
    model.add(Embedding(input_dim=self.embed_matrix.shape[0], output_dim=self.embed_matrix.shape[1], input_length=MAX_LEN, weights=[self.embed_matrix], trainable=False))

    #model.add(LSTM(EMBED_DIMS, dropout=0.2, recurrent_dropout=0.2))
    #model.add(Dense(3,activation='softmax'))

    model.add(LSTM(EMBED_DIMS))
    reg_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(self.embed_matrix.shape[0],)))
    reg_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'))

    model.add(Dense(3,activation='softmax'))

    # adam default parameters:  lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0................................
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model


  def compare_models(self, h1, h2):
    loss_base_model = h1.history['val_loss']
    loss_model = h2.history['val_loss']

    e = range(1, EPOCHS + 1)

    plt.plot(e, loss_base_model, 'bo', label='Validation Loss Model1')
    plt.plot(e, loss_model, 'b', label='Validation Loss Model2')
    plt.legend()
    plt.show()
    plt.savefig("saved_models/compare_models.png")

  def get_weight(self, y):
    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current

  def predict_single_text(self, text):
    model = load_model("saved_models/{}".format(MODEL_FILE))
    with open("saved_models/{}".format(VECTORIZER_FILE), 'rb') as f2:
      vect = pickle.load(f2)
    sequences = vect.texts_to_sequences([text])
    X_test = pad_sequences(sequences, maxlen=MAX_LEN)
    print('predict...')
    pred = model.predict(X_test)[0]
    prob_map = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    print('****************')
    print(prob_map[np.argmax(pred)])
    print('****************')


  def text_to_word_embeddings(self, texts, vocab_size):
    self.tokenizer = Tokenizer(num_words=vocab_size)
    self.tokenizer.fit_on_texts(texts)

    sequences = self.tokenizer.texts_to_sequences(texts)

    x_train = pad_sequences(sequences, maxlen=MAX_LEN)

    with open("saved_models/{}".format(VECTORIZER_FILE), 'wb') as handle:
      pickle.dump(self.tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
      print ('tokenizer saved')

    return x_train

  def is_not_ascii(self, string):
    return string is not None and any([ord(s) >= 128 for s in string])

  def plot_performance(self, history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)
    date_time = 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    plt.title('Model Accuracy\n'+ date_time, fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time, fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()

  def metrics(self):
    pos_count = self.df[self.df.airline_sentiment == 'positive']['tweet_id'].count()
    neg_count = self.df[self.df.airline_sentiment == 'negative']['tweet_id'].count()
    neut_count = self.df[self.df.airline_sentiment == 'neutral']['tweet_id'].count()

    print("tot: {}, pos: {}, neutr:{}, neg: {}".format(self.df.shape[0], pos_count, neut_count, neg_count))
    print('max sentence length', MAX_LEN)

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

    # convert arent' to are not
    words = text.split()
    words = [self.abbr[word] if word in self.abbr else word for word in words]
    text = " ".join(words)

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
# a.metrics()

a.train()
# a.predict_single_text("It's a disgrace!")

# # debug Word2VecCreator
# docs = ['the cat sat on the bench', 'anarchism originated as a term of abuse']
# wv.train(docs)


## LATEST SCORE
# Epoch 00010: val_acc did not improve
# score: 0.58
# acc: 0.81

class AirlineSentimentPredict:
  def __init__(self, tp, filename, col_name):
    self.model = load_model("saved_models/{}".format(MODEL_FILE))
    with open("saved_models/{}".format(VECTORIZER_FILE), 'rb') as f2:
      self.vect = pickle.load(f2)

    self.df = pd.read_csv(filename)
    self.df['max_len'] = self.df[col_name].apply(lambda x: len(x))

    self.col_name = col_name
    self.df['clean_text'] = tp.pre_process(self.df[col_name])

    #self.df['max_len'].max()

  def predict(self):
    sequences = self.vect.texts_to_sequences(self.df['clean_text'].values)
    X_test = pad_sequences(sequences, maxlen=MAX_LEN)
    print('predict...')

    preds = self.model.predict(X_test)
    y_preds = [self.prob_to_sentiment_label(pred) for pred in preds]

    prob_map = ['negative', 'neutral', 'positive']

    probs = []
    for pred in preds:
      di = {}
      for i, prob in enumerate(pred):
        di[prob_map[i]] = prob
      probs.append(di)

    ##probs = ["{}:{}".format(prob_map[i[0]], prob) for i, prob in enumerate(preds)]

    self.df['pred'] = y_preds
    self.df['prob'] = probs

    submission = self.df[[self.col_name, 'pred', 'prob']]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submission.to_csv("predictions-{}.csv".format(timestr))

  def prob_to_sentiment_label(self, pred):
    #THRESHOLD = .4
    #return 0 if pred[0] > THRESHOLD else 1

    return np.argmax(pred)

# p = AirlineSentimentPredict(tp, 'data/test.csv', 'Snippet')
# p.predict()
