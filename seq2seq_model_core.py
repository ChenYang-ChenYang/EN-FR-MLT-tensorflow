# # Language Translation
# In this project, I am going to build language translation model called `seq2seq model or encoder-decoder model` in TensorFlow. The objective of the model is translating English sentences to French sentences. I am going to show the detailed steps, and they will answer to the questions like `how to preprocess the dataset`, `how to define inputs`, `how to define encoder model`, `how to define decoder model`, `how to build the entire seq2seq model`, `how to calculate the loss and clip gradients`, and `how to train and get prediction`.
# 
# This is a part of Udacity's Deep Learning Nanodegree. Some codes/functions (save, load, measuring accuracy, etc) are provided by Udacity. However, majority part is implemented by myself along with much richer explanations and references on each section.  
# 
# ## Get the Data
# While I am running this project on my labtop computer, I cannot handle a huge dataset. Rather I am going to use the reduced size of the original dataset ([WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar)). This version of data is provided by Udacity. If you have GPU machines, feel free to run the codes. There is no other configuration needed.

# In[1]:


import os
import pickle
import copy
import numpy as np
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data


# ## Preprocessing
# 
# Here are brief overview what steps will be done in this section
# 
# - **create lookup tables** 
#   - create two mapping tables 
#       - (key, value) == (unique word string, its unique index)     - `(1)`
#       - (key, value) == (its unique index, unique word string)     - `(2)`
#       - `(1)` is used in the next step, and (2) is used later for prediction step
#       
#       
# - **text to word ids**
#   - convert each string word in the list of sentences to the index
#   - `(1)` is used for converting process
#   
#   
# - **save the pre-processed data**
#   - create two `(1)` mapping tables for English and French
#   - using the mapping tables, replace strings in the original source and target dataset with indicies
# 
# ### Create Lookup Tables
# 
# As mentioned breifly, I am going to implement a function to create lookup tables. Since every models are mathmatically represented, the input and the output(prediction) should also be represented as numbers. That is why this step is necessary for NLP problem because human readable text is not machine readable. This function takes a list of sentences and returns two mapping tables (dictionary data type). Along with the list of sentences, there are special tokens, `<PAD>`, `<EOS>`, `<UNK>`, and `<GO>` to be added in the mapping tables. 
# 
# - (key, value) == (unique word string, its unique index)     - `(1)`
# - (key, value) == (its unique index, unique word string)     - `(2)`
# 
# `(1)` will be used in the next step, `test to word ids`, to find a match between word and its index. `(2)` is not used in pre-processing step, but `(2)` will be used later. After making a prediction, the sequences of words in the output sentence will be represented as their indicies. The predicted output is machine readable but not human readable. That is why we need `(2)` to convert each indicies of words back into human readable words in string.
# 
# <br/>
# <img src='./lookup.png' alt='Drawing' width='70%'>
# 
# #### References
# - [Why special tokens?](https://datascience.stackexchange.com/questions/26947/why-do-we-need-to-add-start-s-end-s-symbols-when-using-recurrent-neural-n)
# - [Python `enumerate`](https://docs.python.org/3/library/functions.html#enumerate)

# In[4]:


def create_lookup_tables(text):
    CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
    # make a list of unique words
    vocab = set(text.split())

    # (1)
    # starts with the special tokens
    vocab_to_int = copy.copy(CODES)

    # the index (v_i) will starts from 4 (the 2nd arg in enumerate() specifies the starting index)
    # since vocab_to_int already contains special tokens
    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    # (2)
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


# ### Text to Word Ids
# 
# Two `(1)` lookup tables will be provided in `text_to_ids` functions as arguments. They will be used in the converting process for English(source) and French(target) respectively. This part is more like a programming part, so there are not much to mention. I will just go over few minor things to remember before jumping in.
# 
# - original(raw) source & target datas contain a list of sentences
#   - they are represented as a string 
# 
# - the number of sentences are the same for English and French
#  
# - by accessing each sentences, need to convert word into the corresponding index.
#   - each word should be stored in a list
#   - this makes the resuling list as a 2-D array ( row: sentence, column: word index )
#   
# - for every target sentences, special token, `<EOS>` should be inserted at the end
#   - this token suggests when to stop creating a sequence
#   
# <br/>
# <img src='./conversion.png' alt='Drawing' width='70%'>
# <br/>

# In[5]:


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
        1st, 2nd args: raw string text to be converted
        3rd, 4th args: lookup tables for 1st and 2nd args respectively
    
        return: A tuple of lists (source_id_text, target_id_text) converted
    """
    # empty list of converted sentences
    source_text_id = []
    target_text_id = []
    
    # make a list of sentences (extraction)
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")
    
    max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
    max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])
    
    # iterating through each sentences (# of sentences in source&target is the same)
    for i in range(len(source_sentences)):
        # extract sentences one by one
        source_sentence = source_sentences[i]
        target_sentence = target_sentences[i]
        
        # make a list of tokens/words (extraction) from the chosen sentence
        source_tokens = source_sentence.split(" ")
        target_tokens = target_sentence.split(" ")
        
        # empty list of converted words to index in the chosen sentence
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
            if (token != ""):
                source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
            if (token != ""):
                target_token_id.append(target_vocab_to_int[token])
                
        # put <EOS> token at the end of the chosen target sentence
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])
            
        # add each converted sentences in the final list
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)
    
    return source_text_id, target_text_id


# ### Preprocess and Save Data
# 
# `create_lookup_tables`, `text_to_ids` are generalized functions. It can  be used for other languages too. In this particular project, the target languages are English and French, so those languages have to fed into `create_lookup_tables`, `text_to_ids` functions to generate pre-processed dataset for this project. Here is the steps to do it.
# 
# - Load data(text) from the original file for English and French
# - Make them lower case letters
# - Create lookup tables for both English and French
# - Convert the original data into the list of sentences whose words are represented in index
# - Finally, save the preprocessed data to the external file (checkpoint)

# In[6]:


def preprocess_and_save_data(source_path, target_path, text_to_ids):
    # Preprocess
    
    # load original data (English, French)
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    # to the lower case
    source_text = source_text.lower()
    target_text = target_text.lower()

    # create lookup tables for English and French data
    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    # create list of sentences whose words are represented in index
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save data for later use
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))


# # Check Point
#  This project uses a small set of sentences. However, in general, NLP requires a huge amount of raw text data. It would take quite a long time to preprocess, so it is recommended to avoid whenever possible. In practice, save the preprocessed data to the external file could speed up your job and let you focus more on building a model.

# In[8]:

def load_preprocess():
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


def check_tf_version_and_gpu():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# ## Build the Neural Network
# 
#  In this notebook, I am going to build a special kind of model called 'sequence to sequence' (seq2seq in short). You can separate the entire model into 2 small sub-models. The first sub-model is called as __[E]__ Encoder, and the second sub-model is called as __[D]__ Decoder. __[E]__ takes a raw input text data just like any other RNN architectures do. At the end, __[E]__ outputs a neural representation. This is a very typical work, but you need to pay attention what this output really is. The output of __[E]__ is going to be the input data for __[D]__.
# 
# That is why we call __[E]__ as Encoder and __[D]__ as Decoder. __[E]__ makes an output encoded in neural representational form, and we don't know what it really is. It is somewhat encrypted. __[D]__ has the ability to look inside the __[E]__'s output, and it will create a totally different output data (translated in French in this case). 
# 
# In order to build such a model, there are 6 steps to do overall. I noted what functions to be implemented are related to those steps.
# - __(1)__ define input parameters to the encoder model
#   - `enc_dec_model_inputs`
# - __(2)__ build encoder model
#   - `encoding_layer`
# - __(3)__ define input parameters to the decoder model
#   - `enc_dec_model_inputs`, `process_decoder_input`, `decoding_layer`
# - __(4)__ build decoder model for training
#   - `decoding_layer_train`
# - __(5)__ build decoder model for inference
#   - `decoding_layer_infer`
# - __(6)__ put (4) and (5) together 
#   - `decoding_layer`
# - __(7)__ connect encoder and decoder models
#   - `seq2seq_model`
# - __(8)__ train and estimate loss and accuracy
# 
# <img src="./training_phase.png" style="width:400px;"/>
# <div style="text-align:center;">Fig 1. Neural Machine Translation / Training Phase</div>
# <br/>
# The figure above is borrowed from Thang Luong's thesis ['Neural Machine Translation'](https://github.com/lmthang/thesis/blob/master/thesis.pdf)

# ### Input (1), (3)
# 
# `enc_dec_model_inputs` function creates and returns parameters (TF placeholders) related to building model. 
# - inputs placeholder will be fed with English sentence data, and its shape is `[None, None]`. The first `None` means the batch size, and the batch size is unknown since user can set it. The second `None` means the lengths of sentences. The maximum length of setence is different from batch to batch, so it cannot be set with the exact number. 
#   - One option is to set the lengths of every sentences to the maximum length across all sentences in every batch. No matter which method you choose, you need to add special character, `<PAD>` in empty positions. However, with the latter option, there could be unnecessarily more `<PAD>` characters.
#   
# 
# - targets placeholder is similar to inputs placeholder except that it will be fed with French sentence data.
# 
# 
# - target_sequence_length placeholder represents the lengths of each sentences, so the shape is `None`, a column tensor, which is the same number to the batch size. This particular value is required as an argument of TrainerHelper to build decoder model for training. We will see in (4).
# 
# 
# - max_target_len gets the maximum value out of lengths of all the target sentences(sequences). As you know, we have the lengths of all the sentences in target_sequence_length parameter. The way to get the maximum value from it is to use [tf.reduce_max](https://www.tensorflow.org/api_docs/python/tf/reduce_max). 

# In[11]:


def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len


# `hyperparam_inputs` function creates and returns parameters (TF placeholders) related to hyper-parameters to the model. 
# - lr_rate is learning rate
# - keep_prob is the keep probability for Dropouts
# 

# In[12]:


def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return lr_rate, keep_prob


# ### Process Decoder Input (3)
# <br/>
# <img src="./go_insert.png" style="width:600px;"/>
# <div style="text-align:center;">Fig 2. `<GO>` insertion</div>
# <br/>
# 
# On the decoder side, we need two different kinds of input for training and inference purposes repectively. While training phase, the input is provided as target label, but they still need to be embeded. On the inference phase, however, the output of each time step will be the input for the next time step. They also need to be embeded and embedding vector should be shared between two different phases.
# 
# In this section, I am going to preprocess the target label data for the training phase. It is nothing special task. What all you need to do is add `<GO>` special token in front of all target data. `<GO>` token is a kind of guide token as saying like "this is the start of the translation". For this process, you need to know three libraries from TensorFlow.
# - [TF strided_slice](https://www.tensorflow.org/api_docs/python/tf/strided_slice)
#   - extracts a strided slice of a tensor (generalized python array indexing).
#   - can be thought as splitting into multiple tensors with the striding window size from begin to end
#   - arguments: TF Tensor, Begin, End, Strides
# - [TF fill](https://www.tensorflow.org/api_docs/python/tf/concat)
#   - creates a tensor filled with a scalar value.
#   - arguments: TF Tensor (must be int32/int64), value to fill
# - [TF concat](https://www.tensorflow.org/api_docs/python/tf/fill)
#   - concatenates tensors along one dimension.
#   - arguments: a list of TF Tensor (tf.fill and after_slice in this case), axis=1
#     
# After preprocessing the target label data, we will embed it later when implementing decoding_layer function.

# In[13]:


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat


# ### Encoding (2)
# 
# <img src="./encoding_model.png" style="width:400px;"/>
# <div style="text-align:center;">Fig 3. Encoding model highlighted - Embedding/RNN layers</div>
# <br/>
# 
# As depicted in Fig 3, the encoding model consists of two different parts. The first part is the embedding layer. Each word in a sentence will be represented with the number of features specified as `encoding_embedding_size`. This layer gives much richer representative power for the words [useful explanation](https://stackoverflow.com/questions/40784656/tf-contrib-layers-embed-sequence-is-for-what/44280918#44280918). The second part is the RNN layer(s). You can make use of any kind of RNN related techniques or algorithms. For example, in this project, multiple LSTM cells are stacked together after dropout technique is applied. You can use different kinds of RNN cells such as GRU.
# 
# Embedding layer
# - [TF contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence)
# 
# RNN layers
# - [TF contrib.rnn.LSTMCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell)
#   - simply specifies how many internal units it has
# - [TF contrib.rnn.DropoutWrapper](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper)
#   - wraps a cell with keep probability value 
# - [TF contrib.rnn.MultiRNNCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell)
#   - stacks multiple RNN (type) cells
#   - [how this API is used in action?](https://github.com/tensorflow/tensorflow/blob/6947f65a374ebf29e74bb71e36fd82760056d82c/tensorflow/docs_src/tutorials/recurrent.md#stacking-multiple-lstms)
#   
# Encoding model
# - [TF nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
#   - put Embedding layer and RNN layer(s) all together

# In[14]:


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)
    
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
    
    outputs, state = tf.nn.dynamic_rnn(stacked_cells, 
                                       embed, 
                                       dtype=tf.float32)
    return outputs, state


# ### Decoding - Training process (4)
# 
# Decoding model can be thought of two separate processes, training and inference. It is not they have different architecture, but they share the same architecture and its parameters. It is that they have different strategy to feed the shared model.
# 
# For this(training) and the next(inference) section, Fig 4 shows clearly shows what they are.
# 
# <img src="./decoder_shift.png" style="width:700px;"/>
# <div style="text-align:center;">Fig 4. Decoder shifted inputs</div>
# <br/>
# 
# While encoder uses [TF contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence), it is not applicable to decoder even though it may require its input embeded. That is because the same embedding vector should be shared via training and inferece phases. [TF contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence) can only embed the prepared dataset before running. What needed for inference process is dynamic embedding capability. It is impossible to embed the output from the inference process before running the model because the output of the current time step will be the input of the next time step.
# 
# How we can embed? We will see soon. However, for now, what you need to remember is training and inference processes share the same embedding parameters. For the training part, embeded input should be delivered. On the inference part, only embedding parameters used in the training part should be delivered.
# 
# Let's see the training part first. 
# - [`tf.contrib.seq2seq.TrainingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper)
#   - TrainingHelper is where we pass the embeded input. As the name indicates, this is only a helper instance. This instance should be delivered to the BasicDecoder, which is the actual process of building the decoder model.
# - [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
#   - BasicDecoder builds the decoder model. It means it connects the RNN layer(s) on the decoder side and the input prepared by TrainingHelper.
# - [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)
#   - dynamic_decode unrolls the decoder model so that actual prediction can be retrieved by BasicDecoder for each time steps.

# In[15]:


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer 
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                               target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_summary_length)
    return outputs


# ### Decoding - Inference process (5)
# 
# - [`tf.contrib.seq2seq.GreedyEmbeddingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper)
#   - GreedyEmbeddingHelper dynamically takes the output of the current step and give it to the next time step's input. In order to embed the each input result dynamically, embedding parameter(just bunch of weight values) should be provided. Along with it, GreedyEmbeddingHelper asks to give the `start_of_sequence_id` for the same amount as the batch size and `end_of_sequence_id`.
# - [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
#   - same as described in the training process section
# - [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)
#   - same as described in the training process section

# In[16]:


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer 
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], start_of_sequence_id), 
                                                      end_of_sequence_id)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)
    
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


# ### Build the Decoding Layer (3), (6)
# 
# __Embed the target sequences__
# 
# - [TF contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence) creates internal representation of embedding parameter, so we cannot look into or retrieve it. Rather, you need to create a embedding parameter manually by [TF Variable](https://www.tensorflow.org/api_docs/python/tf/Variable). 
# 
# - Manually created embedding parameter is used for training phase to convert provided target data(sequence of sentence) by [TF nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup) before the training is run. [TF nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup) with manually created embedding parameters returns the similar result to the [TF contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence). For the inference process, whenever the output of the current time step is calculated via decoder, it will be embeded by the shared embedding parameter and become the input for the next time step. You only need to provide the embedding parameter to the GreedyEmbeddingHelper, then it will help the process.
# 
# - [How embedding_lookup works?](https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do)
#   - In short, it selects specified rows
#   
# - Note: Please be careful about setting the variable scope. As mentioned previously, parameters/variables are shared between training and inference processes. Sharing can be specified via [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope).
# 
# __Construct the decoder RNN layer(s)__
# - As depicted in Fig 3 and Fig 4, the number of RNN layer in the decoder model has to be equal to the number of RNN layer(s) in the encoder model.
# 
# __Create an output layer to map the outputs of the decoder to the elements of our vocabulary__
# - This is just a fully connected layer to get probabilities of occurance of each words at the end.

# In[17]:


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output)


# ### Build the Seq2Seq model (7)
# 
# In this section, previously defined functions, `encoding_layer`, `process_decoder_input`, and `decoding_layer` are put together to build the big picture, Sequence to Sequence model. 

# In[18]:


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output = decoding_layer(dec_input,
                                               enc_states, 
                                               target_sequence_length, 
                                               max_target_sentence_length,
                                               rnn_size,
                                              num_layers,
                                              target_vocab_to_int,
                                              target_vocab_size,
                                              batch_size,
                                              keep_prob,
                                              dec_embedding_size)
    
    return train_output, infer_output


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


# ### Train
# 
# `get_accuracy`
# - compare the lengths of target(label) and logits(prediction)
# - add(pad) 0s at the end of the ones having the shorter length
#   - `[(0,0),(0,max_seq - target.shape[1])]` indicates the 2D array. The first (0,0) means no padding for the first dimension. The second (0, ...) means there is no pads in front of the second dimension but pads at the end. And pad as many times as ... .
# - above process is to makes two entities to have the same shape (length)
# - finally, returns the average of where the target and logits have the same value (1)
# 
# [numpy pad function](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html)

# In[22]:


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))


# ### Save Parameters
# Save the `batch_size` and `save_path` parameters for inference.

# In[ ]:


def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)


def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results





