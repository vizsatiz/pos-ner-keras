from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


class Network:

    def __init__(self,
                 sentences,
                 tags):
        self.input_shape = None
        self.model = None
        self.word2index = None
        self.tag2index = None
        self.train_sentences = None
        self.test_sentences = None
        self.train_tags = None
        self.test_tags = None
        self.sentences = sentences
        self.tags = tags
        self.train_sentences_x, self.test_sentences_x, self.train_tags_y, self.test_tags_y = [], [], [], []

    def init(self, ignore_class_accuracy):
        self.__split_data_for_training()
        max_len = len(max(self.train_sentences_x, key=len))
        self.input_shape = max_len
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.input_shape,)))
        self.model.add(Embedding(len(self.word2index), 128))
        self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(len(self.tag2index))))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(0.001),
                           metrics=['accuracy', ignore_class_accuracy(0)])
        self.model.summary()

    def train(self):
        self.model.fit(self.train_sentences_x, to_categorical(self.train_tags_y, len(self.tag2index)),
                       batch_size=128,
                       epochs=40,
                       validation_split=0.2)
        scores = self.model.evaluate(self.test_sentences_x, to_categorical(self.test_tags_y, len(self.tag2index)))
        print(f"{self.model.metrics_names[1]}: {scores[1] * 100}")

    def __split_data_for_training(self):
        (self.train_sentences,
         self.test_sentences,
         self.train_tags,
         self.test_tags) = train_test_split(self.sentences, self.tags, test_size=0.2)
        self.__set_up_padding_and_oov()
        for s in self.train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.word2index[w.lower()])
                except KeyError:
                    s_int.append(self.word2index['-OOV-'])

            self.train_sentences_x.append(s_int)
        for s in self.test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.word2index[w.lower()])
                except KeyError:
                    s_int.append(self.word2index['-OOV-'])

            self.test_sentences_x.append(s_int)
        for s in self.train_tags:
            self.train_tags_y.append([self.tag2index[t] for t in s])

        for s in self.test_tags:
            self.test_tags_y.append([self.tag2index[t] for t in s])

        self.train_sentences_x = pad_sequences(self.train_sentences_x, maxlen=self.input_shape, padding='post')
        self.test_sentences_x = pad_sequences(self.test_sentences_x, maxlen=self.input_shape, padding='post')
        self.train_tags_y = pad_sequences(self.train_tags_y, maxlen=self.input_shape, padding='post')
        self.test_tags_y = pad_sequences(self.test_tags_y, maxlen=self.input_shape, padding='post')

    def __set_up_padding_and_oov(self):
        words, tags = set([]), set([])

        for s in self.train_sentences:
            for w in s:
                words.add(w.lower())

        for ts in self.train_tags:
            for t in ts:
                tags.add(t)

        word2index = {w: i + 2 for i, w in enumerate(list(words))}
        word2index['-PAD-'] = 0  # The special value used for padding
        word2index['-OOV-'] = 1  # The special value used for OOVs

        tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        tag2index['-PAD-'] = 0  # The special value used to padding
