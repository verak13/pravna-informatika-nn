import numpy as np
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras
from keras.models import Model, load_model
import load_dataset as ld

# configuration
# for training
BATCH_SIZE = 64
EPOCHS = 10
# dimension of hidden vector (encoder)
LATENT_DIM = 256
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
# for save
BASE_PATH = 's2s-transfer-test-10-10-'

# load the data
# original text
input_texts = []
# summarized text
target_texts = []
# with the start of sentence token
target_texts_inputs = []

# max_len_input = 365

# find max len input
in_length_list = []
out_length_list = []
target_len_list = []
for i in range(0, 235, 10):
    temp_list = []
    target_texts = []
    target_texts_inputs = []
    data = gd.read_training_dataset_optimized(i, i+10)
    for i in data:
        input_text, summarization = i['original'], i['summary']
        # make the target input and output
        target_text = summarization + ' <eos>'
        # used for teacher forcing
        target_texts_input = '<sos> ' + summarization
        temp_list.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_texts_input)
    tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer_inputs.fit_on_texts(temp_list)
    input_sequences = tokenizer_inputs.texts_to_sequences(temp_list)
    current_max = max(len(s) for s in input_sequences)
    print(current_max)
    word2idx_inputs = tokenizer_inputs.word_index
    print('Found %s unique output tokens' % len(word2idx_inputs))
    in_length_list.append(len(word2idx_inputs))

    tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
    tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)  # inefficient, oh well
    target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
    target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

    # get the word to index mapping for output
    word2idx_outputs = tokenizer_outputs.word_index
    print('Found %s unique output tokens' % len(word2idx_outputs))
    out_length_list.append(len(word2idx_outputs))
    target_len_list.append(max(len(s) for s in target_sequences))

max_len_input = max(in_length_list)
num_words_output = max(out_length_list) + 1
max_len_target = max(target_len_list)
print('Max len target sequence:', max_len_target)
print('Found max_num_v to be ', max_len_input)
print('Num words output: ', num_words_output)

# load in the data
# f = open('docs/data.json', encoding='utf-8')
# data = json.load(f)
ranges = [[x, x+15] for x in range(0, 235, 15)]
INDEX_OF_EXPERIMENT = 1
for data_range in ranges:
    print('------------------------------------')
    print('Current elements - ', (INDEX_OF_EXPERIMENT-1)*15, ' - ', (INDEX_OF_EXPERIMENT)*15)
    data = ld.read_training_dataset_optimized(*data_range)
    input_texts = []
    target_texts = []
    target_texts_inputs = []

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
    input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
    # get the word to index mapping for input
    word2idx_inputs = tokenizer_inputs.word_index
    print('Found %s unique input tokens' % len(word2idx_inputs))

    # determine maximum length in input sequence
    # max_len_input = max(len(s) for s in input_sequences)
    # print('Max len input sequence:', max_len_input)

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
    # + 1 because indexing starts at 1 -------------------------------
    # num_words_output = len(word2idx_outputs) + 1

    # determine maximum length of output sequence
    # max_len_target = max(len(s) for s in target_sequences)
    # print('Max len target sequence:', max_len_target)

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
    num_words = max_len_input + 1
    embedding_layer = Embedding(
        num_words,  # size of the vocabulary
        EMBEDDING_DIM,  # length of the vector for each word
        input_length=MAX_SEQUENCE_LENGTH,
        embeddings_initializer='uniform',
        trainable=False
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


    # [<KerasTensor: shape=(None, 256) dtype=float32 (created by layer 'lstm')>, <KerasTensor: shape=(None, 256) dtype=float32 (created by layer 'lstm')>]

    if INDEX_OF_EXPERIMENT != 1:
        previous_model = load_model(BASE_PATH + str(INDEX_OF_EXPERIMENT-1) + '-00000010.h5')

    # the encoder will be stand-alone
    # from this we will get our initial decoder hidden state
    encoder_inputs_placeholder = Input(shape=(max_len_input,))
    # encoder_model = Model(previous_model.layers[0].output, encoder_states)
    x = embedding_layer(encoder_inputs_placeholder)

    # encoder_outputs_2, state_h2, state_c2 = previous_model.layers[4].output
    # encoder_states = [state_h2, state_c2]
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

    decoder_dense = Dense(num_words_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)



    model = Model(
        [encoder_inputs_placeholder, decoder_inputs_placeholder],
        decoder_outputs
    )
    # if INDEX_OF_EXPERIMENT != 1:
    #     for (layer, p_model_l) in zip(model.layers, previous_model.layers):
    #         weights = p_model_l.get_weights()
    #         print (layer.get_config())
    #         print (p_model_l.get_config())
    #
    #         if len(weights) != 0:
    #             print('SETWEIGHTS - ', weights)
    #             print('WEIGHTS - ', layer.get_weights())
    #             layer.set_weights(weights)
    #             print('set some weights')

    # model.set_weights(previous_model.get_weights())
    # model = previous_model
    print('compiling model...')

    model.trainable = True
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True,

    )
    print('compiling model done')
    checkpoint = keras.callbacks.ModelCheckpoint(BASE_PATH + str(INDEX_OF_EXPERIMENT) + '-{epoch:08d}.h5', period=10)
    r = model.fit(
        [encoder_inputs, decoder_inputs],
        decoder_target_one_hot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=2,
        callbacks=[checkpoint]
    )
    INDEX_OF_EXPERIMENT += 1

