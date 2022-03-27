import numpy as np
import pickle
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras
import load_dataset as ld


# configuration
BATCH_SIZE = 64  # batch size for training
EPOCHS = 5000  # number of epochs to train for
LATENT_DIM = 256  # latent dimensionality of the encoding space
NUM_SAMPLES = 10000  # number of samples to train on
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# load the data
# original text
input_texts = []
# summarized text
target_texts = []
# with the start of sentence token
target_texts_inputs = []


# load in the data
# f = open('docs/data.json', encoding='utf-8')
# data = json.load(f)
data = ld.read_training_dataset(0, 25)


# Iterating through the json list
for i in data:
    input_text, summarization = i['original'], i['summary']
    # make the target input and output
    target_text = summarization + ' <eos>'
    # used for teacher forcing
    target_texts_input = '<sos> ' + summarization

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_texts_input)

print('Num samples:', len(input_texts))

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
# pickle tokenizer_inputs
with open('pickles-for-testing/tokenizer.pkl', 'wb') as pkl_handle:
    pickle.dump(tokenizer_inputs, pkl_handle)

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens' % len(word2idx_inputs))
with open('pickles-for-testing/word2idx_inputs.pkl', 'wb') as pkl_handle:
    pickle.dump(word2idx_inputs, pkl_handle)

# determine maximum length in input sequence
max_len_input = max(len(s) for s in input_sequences)
print('Max len input sequence:', max_len_input)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)  # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens' % len(word2idx_outputs))
with open('pickles-for-testing/word2idx_outputs.pkl', 'wb') as pkl_handle:
    pickle.dump(word2idx_outputs, pkl_handle)

# number of outputs for later
# + 1 because indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length of output sequence
max_len_target = max(len(s) for s in target_sequences)
print('Max len target sequence:', max_len_target)

# pad the sequences
# add zeros at the beginning
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print('encoder_data.shape:', encoder_inputs.shape)
# print('encoder_data[0]:', encoder_inputs[0])
# add zeros at the end
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print('encoder_data.shape:', decoder_inputs.shape)
# print('encoder_data[0]:', decoder_inputs[0])
# add zeros at the end
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# in that case set trainable on False
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_layer = Embedding(
    num_words,  # size of the vocabulary
    EMBEDDING_DIM,  # length of the vector for each word
    input_length=MAX_SEQUENCE_LENGTH,
    embeddings_initializer='uniform',
    trainable=True
)
print('embedding done')

# create targets
decoder_target_one_hot = np.zeros(
    (
        len(input_texts),
        max_len_target,
        num_words_output
    ),
    dtype='float32'
)

print('one hot done')
# assign the values
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        decoder_target_one_hot[i, t, word] = 1

# build the model
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)
encoder_outputs, h, c = encoder(x)

# keep the states to pass into decoder
encoder_states = [h, c]

# set up the decoder, using [h, c] as initial state
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# since the decoder is a "to-many" model we want to have:
# return_sequences = True
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs_x,
    initial_state=encoder_states
)

# final dense layer for predictions
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
print('create model')

model = Model(
    [encoder_inputs_placeholder, decoder_inputs_placeholder],
    decoder_outputs
)
print('compiling model...')

# for i, v in enumerate(model.layers):
#     print(i, v)
# 0 <keras.engine.input_layer.InputLayer object at 0x0000018D6F2BA830>
# 1 <keras.engine.input_layer.InputLayer object at 0x0000018D6F2BAFE0>
# 2 <keras.layers.embeddings.Embedding object at 0x0000018D6F1335B0>
# 3 <keras.layers.embeddings.Embedding object at 0x0000018D6F2BAF80>
# 4 <keras.layers.recurrent_v2.LSTM object at 0x0000018D6F2BB040>
# 5 <keras.layers.recurrent_v2.LSTM object at 0x0000018E38749030>
# 6 <keras.layers.core.dense.Dense object at 0x0000018D6F292320>

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    run_eagerly=True
)

print('compiling model done')
checkpoint = keras.callbacks.ModelCheckpoint('s2s-{epoch:08d}.h5', period=100)
r = model.fit(
    [encoder_inputs, decoder_inputs],
    decoder_target_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=2,
    callbacks=[checkpoint]
)


print('fit model done')
model.save('s2s-final-5000.h5')

# make predictions
# another model that can take in the rnn state and previous
# word as input and accept a T = 1 sequence

# the encoder will be stand-alone
# from this we will get our initial decoder state
encoder_model = Model(encoder_inputs_placeholder, encoder_states)
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# keep the states too, to be output of the sampling model
decoder_outputs, h, c = decoder_lstm(
    decoder_inputs_single_x,
    initial_state=decoder_states_inputs
)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)
# the sampling model
# inputs: y(t-1), h(t-1), c(t-1)
# outputs: y(t), h(t), c(t)
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# map indexes back so we can view the result
idx2word_i = {v: k for k, v in word2idx_inputs.items()}
idx2word_o = {v: k for k, v in word2idx_outputs.items()}
# print('>>>>')
# print(idx2word_i)
# print('>>>>')
# print(idx2word_o)


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
        # get next word
        idx = np.argmax(output_tokens[0, 0, :])
        # end sentence of eos
        if eos == idx:
            break
        word = ''
        if idx > 0:
            word = idx2word_o[idx]
            output_sentence.append(word)

        # update the decoder input
        # which is the word we just generated
        target_seq[0, 0] = idx

        # update states
        states_value = [h, c]
    return ' '.join(output_sentence)


from rouge import Rouge
rouge = Rouge()

of = open('results_s2s.txt', 'w', encoding='utf-8')
for i in range(len(input_texts)):
    input_seq = encoder_inputs[i:i+1]
    summarization = decode_sequence(input_seq)
    print('-----')
    print('Input:', input_texts[i])
    print('Summarization:', summarization)
    actual_summary = target_texts[i].replace("<eos>", "")
    print(rouge.get_scores(summarization.lower() + ' test', actual_summary.lower()))
    of.write('-----')
    of.write('\n')
    of.write('Input:' + input_texts[i])
    of.write('\n')
    of.write('Summarization:' + summarization)
    of.write('\n')
    of.write('Rouge score: ' + str(rouge.get_scores(summarization.lower() + ' test', actual_summary.lower())))
    of.write('\n')
of.close()
