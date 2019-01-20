import tensorflow as tf
import seq2seq_model_core

if __name__ == '__main__':

    display_step = 300

    epochs = 13
    batch_size = 128

    rnn_size = 128
    num_layers = 3

    encoding_embedding_size = 200
    decoding_embedding_size = 200

    learning_rate = 0.001
    keep_probability = 0.5

    _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = seq2seq_model_core.load_preprocess()
    load_path = seq2seq_model_core.load_params()


    left_roll = '花 开 富 贵 家 家 乐'

    left_roll = seq2seq_model_core.sentence_to_seq(left_roll, source_vocab_to_int)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        right_roll = sess.run(logits, {input_data: [left_roll]*batch_size,
                                             target_sequence_length: [len(left_roll)*2]*batch_size,
                                             keep_prob: 1.0})[0]

    print('Input')
    print('  上联 Ids:      {}'.format([i for i in left_roll]))
    print('  上联: {}'.format([source_int_to_vocab[i] for i in left_roll]))

    print('\nPrediction')
    print('  下联 Ids:      {}'.format([i for i in right_roll]))
    print('  下联: {}'.format(" ".join([target_int_to_vocab[i] for i in right_roll])))
