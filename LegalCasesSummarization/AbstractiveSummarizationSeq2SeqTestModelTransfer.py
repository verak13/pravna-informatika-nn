from __future__ import print_function, division

import json
import pickle
from builtins import range, input

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input, LSTM, GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import load_dataset as ld

LATENT_DIM = 256  # latent dimensionality of the encoding space
NUM_SAMPLES = 10000  # number of samples to train on
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100

# load the data
input_texts = [] # sentence in original language
target_texts = [] # sentence in target language
max_len_input = 2183
max_len_target = 30

with open('tokenizer.pkl', 'rb') as pickle_handle:
    tokenizer_inputs = pickle.load(pickle_handle)

# load dictionaries
with open('word2idx_inputs.pkl', 'rb') as pickle_handle:
    word2idx_inputs = pickle.load(pickle_handle)
with open('word2idx_outputs.pkl', 'rb') as pickle_handle:
    word2idx_outputs = pickle.load(pickle_handle)
# print(word2idx_outputs)

num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
num_words_output = len(word2idx_outputs) + 1

# load in the data
data = ld.read_test_dataset()


# Iterating through the json list
for i in data:
    input_text, summarization = i['original'], i['summary']
    # make the target input and output
    target_text = summarization + ' <eos>'
    # used for teacher forcing
    target_texts_input = '<sos> ' + summarization

    input_texts.append(input_text)
    target_texts.append(target_text)

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

embedding_layer = Embedding(
    num_words,  # vocabulary size
    EMBEDDING_DIM,  # embedding dimension
    embeddings_initializer='uniform',
    input_length=MAX_SEQUENCE_LENGTH,  # how long sequences will be
    trainable=True
)

# make predictions
# we need to create another model that can take in
# the rnn state and previous word as input
# and accept a T=1 sequence

# encoder_states = [h, c] ovo treba ucitati iz sacuvanog modela
from_to = [100, 600, 100]
model_num_list = [str(x) for x in range(*from_to)]
model_list =[ 's2s-transfer-1-1-' + x.zfill(8) + '.h5' for x in model_num_list]
# for model_path in model_list:
#     print(model_path)
model_list.extend([ 's2s-transfer-1-2-' + x.zfill(8) + '.h5' for x in model_num_list])

from rouge import Rouge
rouge = Rouge()
for k, test_text in enumerate(input_texts):
    print('--------------------------------------')
    print('Input:', input_texts[k])

    of = open('test-trans-' + str(k) + '.txt', 'w', encoding='utf-8')
    of.write(input_texts[k] + '\n')
    for model_path in model_list:
        # print(model_path)
        new_model = load_model(model_path)

        # for i, v in enumerate(new_model.layers):
        #     print(i, v)
        encoder_outputs_2, state_h2, state_c2 = new_model.layers[4].output
        # print(encoder_outputs_2)
        # print(state_h2)
        # print(state_c2)
        # print("----")
        # encoder_outputs_2, state_h2, state_c2 = new_model.layers[5].output
        # print(encoder_outputs_2)
        # print(state_h2)
        # print(state_c2)
        encoder_states = [state_h2, state_c2]
        # print(encoder_states)
        # print(max_len_input)


        # [<KerasTensor: shape=(None, 256) dtype=float32 (created by layer 'lstm')>, <KerasTensor: shape=(None, 256) dtype=float32 (created by layer 'lstm')>]

        # the encoder will be stand-alone
        # from this we will get our initial decoder hidden state
        encoder_inputs_placeholder = Input(shape=(max_len_input,))
        encoder_model = Model(new_model.layers[0].output, encoder_states)
        decoder_state_input_h = Input(shape=(LATENT_DIM,))
        decoder_state_input_c = Input(shape=(LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_embedding = Embedding(num_words_output, LATENT_DIM)
        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        # this time we want to keep the states too, to be output
        # by our sampling model
        decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.5)
        decoder_outputs, h, c = decoder_lstm(
            decoder_inputs_single_x,
            initial_state=decoder_states_inputs
        )

        decoder_states = [h, c]
        decoder_dense = Dense(num_words_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # the sampling model
        # inputs: y(t-1), h(t-1), c(t-1)
        # outputs: y(t), h(t), c(t)
        decoder_model = Model(
            [decoder_inputs_single] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        # map indexes back into real world
        # so we can view the results
        idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
        idx2word_trans = {v: k for k, v in word2idx_outputs.items()}


        def decode_sequence(input_seq):
            # encode the input as state vectors
            states_value = encoder_model.predict(input_seq)
            # generate empty target sequence of length 1
            target_seq = np.zeros((1, 1))
            # populate the first character of target sequence with the start character
            # NOTE: tokenizer lower-cases all words
            target_seq[0, 0] = word2idx_outputs['<sos>']

            # if we get this we break
            eos = word2idx_outputs['<eos>']

            # create the translation
            output_sentence = []
            for _ in range(max_len_target):
                output_tokens, h, c = decoder_model.predict(
                    [target_seq] + states_value
                )

                # output_tokens, h = decoder_model.predict(
                #     [target_seq] + states_value
                # ) # gru

                # get next word
                idx = np.argmax(output_tokens[0, 0, :])

                # end sentence of eos
                if eos == idx:
                    break

                word = ''
                if idx > 0:
                    word = idx2word_trans[idx]
                    output_sentence.append(word)

                # update the decoder input
                # which is the word we just generated
                target_seq[0, 0] = idx

                # update states
                states_value = [h, c]
                # states_value = [h] # gru
            return ' '.join(output_sentence)


        # while True:
            # do some test translations

            # i = np.random.choice(len(input_texts))
        input_seq = encoder_inputs[k:k+1]
        summarization = decode_sequence(input_seq)
        actual_summary = target_texts[k].replace("<eos>", "")
        r_scores = rouge.get_scores(summarization.lower() + ' test', actual_summary.lower())[0]['rouge-1']
        print(model_path, ' - ', summarization)
        # print('Rouge - f1 ',r_scores.f, ', precision ', r_scores.p, ', recall', r_scores.r)
        cap = max(len(summarization), 5)
        unique_set = list(set(summarization.split(' ')[:cap]))
        unique_set = [x.replace(',', ' ') for x in unique_set]
        unique_set = [x.replace('.', ' ') for x in unique_set]
        print('Unique - ', ','.join(unique_set))
        print('Rouge - ', str(r_scores))

        csv_string = model_path + ', '  +  ', '.join(unique_set) + ', ' + str(r_scores) + '\n'
        of.write(csv_string)
    of.close()
