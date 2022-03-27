import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, RepeatVector, Concatenate, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras.backend as K


# we do softmax over time axis
# expected shape is N x T x D
# N - number of samples
# T - sequence length
# D vector dimensionality
def softmax_over_time(x):
    assert(K.ndim(x) > 2)
    # axis=1 is the time dimension
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


# configuration
# for training
BATCH_SIZE = 64
EPOCHS = 100
# dimension of hidden vector (encoder)
LATENT_DIM = 256
# for testing puproses 100
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
LATENT_DIM_DECODER = 256  # try different to see if it works

# load the data
# original text
input_texts = []
# summarized text
target_texts = []
# with the start of sentence token
target_texts_inputs = []

# load in the data
for line in open('test-attention/srp.txt', encoding='utf-8'):
    # input and target are separated by tab
    if '\t' not in line:
        continue

    # split up the input and translation
    input_text, translation, _ = line.split('\t')

    # make the target input and output
    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)


# f = open('docs/data.json', encoding='utf-8')
# data = json.load(f)
#
# # Iterating through the json list
# for i in data:
#     input_text, summarization = i['original'], i['summary']
#     # make the target input and output
#     target_text = summarization + ' <eos>'
#     # used for teacher forcing
#     target_texts_input = '<sos> ' + summarization
#
#     input_texts.append(input_text)
#     target_texts.append(target_text)
#     target_texts_inputs.append(target_texts_input)
# # Closing file
# f.close()
print('Num samples:', len(input_texts))

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens' % len(word2idx_inputs))

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

# would be nice to load pre-trained word embeddings from bertic?
# in that case set trainable on False
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_layer = Embedding(
    num_words,  # size of the vocabulary
    EMBEDDING_DIM,  # length of the vector for each word
    input_length=MAX_SEQUENCE_LENGTH,
    embeddings_initializer='uniform',
    trainable=True
)

# create targets
decoder_target_one_hot = np.zeros(
    (
        len(input_texts),
        max_len_target,
        num_words_output
    ),
    dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        decoder_target_one_hot[i, t, word] = 1

# build the model
print('Build the model')
encoder_inputs_placeholders = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholders)
encoder = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, dropout=0.5))
encoder_outputs = encoder(x)

# set up the decoder
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

### ATTENTION ###
# attention layers need to be global
# because will be repeated Ty times at the decoder
print('Build attention mechanism')
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1)  # to perform the weighted sum of aplha[t] * h[t]


def one_step_attention(h, st_1):
    print('One step attention')
    # h = h(1), ... , h(Tx), shape = (Tx, LATENT_DIM * 2)
    # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

    # copy s(t-1) Tx times
    # now shape = (Tx, LATENT_DIM_DECODER + LATENT_DIM*2)
    st_1 = attn_repeat_layer(st_1)

    # concatenate all h(t)'s with s(t-1)
    # now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
    x = attn_concat_layer([h, st_1])

    # neural net first layer
    x = attn_dense1(x)

    # neural net second layer with special softmax over time
    alphas = attn_dense2(x)

    # "dot" the alphas and the h's
    # a.dot(b) = sum over a[t] * b[t]
    context = attn_dot([alphas, h])

    return context


# define the rest of the decoder
# after attention
print('Define the rest of the decoder')
decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)
# output word probabilities
decoder_dense = Dense(num_words_output, activation='softmax')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
# this is for teacher forcing
# combines the previous correct word with the context
context_last_word_concat_layer = Concatenate(axis=2)

# unlike previous seq2seq, we cannot get the
# output all in one step
# instead we need to do Ty steps
# and in each of those steps, we need to consider
# all Tx h's

# s, c will be re-assigned in each iteration
# of the loop
s = initial_s
c = initial_c

# collect outputs in a list at first
outputs = []
print('Collect outputs in a list')
for t in range(max_len_target):  # Ty times
    # get the context using attention
    # we do ones step of attention - encoder states, with the previous decoder state
    context = one_step_attention(encoder_outputs, s)

    # we need a different layer for each time step
    # we don't want to concatenate the context with the entire input sequence
    # we need to collect the right slice - we only want the previous correct word
    # first dimension sample, second dimension time - t == time
    # not global because we need each layer to index a different point in time
    selector = Lambda(lambda x: x[:, t:t+1])
    xt = selector(decoder_inputs_x)

    # combine
    # here we concatenate context with the right word
    decoder_lstm_input = context_last_word_concat_layer([context, xt])

    # pass the combined [context, last word] into the lstm
    # along with [s, c]
    # get the new [s, c] and output
    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

    # final dense layer to get the next word prediction
    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)


# 'outputs' is now a list of length Ty
# each element is of shape (batch size (N), output vocab size (D))
# therefore if we simply stack all the outputs into 1 tensor
# it would be of shape T x N x D
# we would like it to be of shape N x T x D
# we need to stack and transpose because we need N on the first place
def stack_and_transpose(x):
    # x is a list of length T, each element is a batch size x output vocab size tensor
    x = K.stack(x)  # is now T x batch size x output vocab size tensor
    x = K.permute_dimensions(x, pattern=(1, 0, 2))  # is now batch size x t x output vocab size
    return x


# make it a layer
stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

model = Model(
    inputs=[
        encoder_inputs_placeholders,
        decoder_inputs_placeholder,
        initial_s,
        initial_c
    ],
    outputs=outputs
)

# compile the model
print('Compile second model')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
# NUM_SAMPLES = 3
z = np.zeros((len(encoder_inputs), LATENT_DIM_DECODER))  # initial [s, c]
print('Fit second model')
r = model.fit(
    [encoder_inputs, decoder_inputs, z, z],
    decoder_target_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)


### make predictions ###
# we need to create another model
# that can take in the RNN state and previous word as input
# and accept
# a T = 1 sequence

# the encoder will be stand alone
# from this we will get our initial decoder hidden state
# i. e. h(1), ..., h(Tx)
encoder_model = Model(encoder_inputs_placeholders, encoder_outputs)

# next we define a T = 1 decoder model
# this are hidden states
encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# no need to loop over attention steps this time
# because there is only one step
context = one_step_attention(encoder_outputs_as_input, initial_s)

# combine context with last word
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

# lstm and final dense
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)

# there is no need for the final stack and transpose
# there's only 1 output
# it is already size N x D

# create the model object
decoder_model = Model(
    inputs=[
        decoder_inputs_single,
        encoder_outputs_as_input,
        initial_s,
        initial_c
    ],
    outputs=[decoder_outputs, s, c]
)

# map indexes back into real words
# so we can view the results
idx2word_i = {v: k for k, v in word2idx_inputs.items()}
idx2word_o = {v: k for k, v in word2idx_outputs.items()}


def decode_sequence(input_seq):
    print('Decode sequence')
    # encode the input as state vectors
    enc_out = encoder_model.predict(input_seq)

    # generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    # populate the first character of target sequence with the start character
    # NOTE: tokenizer lower-cases all words
    target_seq[0, 0] = word2idx_outputs['<sos>']

    # if we get this we break
    eos = word2idx_outputs['<eos>']

    # [s, c] will be updated in each loop iteration
    s = np.zeros((1, LATENT_DIM_DECODER))
    c = np.zeros((1, LATENT_DIM_DECODER))

    # create the translation
    output_sentence = []
    for _ in range(max_len_target):
        o, s, c = decoder_model.predict(
            [target_seq, enc_out, s, c]
        )
        # get next word
        idx = np.argmax(o.flatten())
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
    return ' '.join(output_sentence)


while True:
    # Do some test translations
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i:i+1]
    summary = decode_sequence(input_seq)
    print('-')
    print('Input:', input_texts[i])
    print('Summary:', summary)
    # print('Actual summary:', target_texts[i])

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break

