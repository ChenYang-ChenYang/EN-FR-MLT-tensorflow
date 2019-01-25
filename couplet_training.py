import numpy as np
import seq2seq_model_core
from collections import Counter
import tensorflow as tf
from datetime import datetime

if __name__ == '__main__':

    # load data
    source_path = './data/left_roll'
    target_path = './data/right_roll'
    source_text = seq2seq_model_core.load_data(source_path)
    target_text = seq2seq_model_core.load_data(target_path)


    # explore Data
    print('Dataset Brief Stats')
    print('* number of unique words in left roll: {}        [this is roughly measured/without any preprocessing]'.format(len(Counter(source_text.split()))))
    print()

    left_roll_sentences = source_text.split('\n')
    print('* Left roll sentences')
    print('\t- number of sentences: {}'.format(len(left_roll_sentences)))
    print('\t- avg. number of words in a sentence: {}'.format(np.average([len(sentence.split()) for sentence in left_roll_sentences])))

    right_roll_sentences = target_text.split('\n')
    print('* Right roll sentences')
    print('\t- number of sentences: {} [data integrity check / should have the same number]'.format(len(right_roll_sentences)))
    print('\t- avg. number of words in a sentence: {}'.format(np.average([len(sentence.split()) for sentence in right_roll_sentences])))
    print()

    sample_sentence_range = (0, 5)
    side_by_side_sentences = list(zip(left_roll_sentences, right_roll_sentences))[sample_sentence_range[0]:sample_sentence_range[1]]
    print('* Sample sentences range from {} to {}'.format(sample_sentence_range[0], sample_sentence_range[1]))

    # for index, sentence in enumerate(side_by_side_sentences):
    #     en_sent, fr_sent = sentence
    #     print('[{}-th] sentence'.format(index+1))
    #     print('\tEN: {}'.format(en_sent))
    #     print('\tFR: {}'.format(fr_sent))
    #     print()


    # preprocess and save data
    seq2seq_model_core.preprocess_and_save_data(source_path, target_path, seq2seq_model_core.text_to_ids)

    # check the Version of TensorFlow and Access to GPU
    seq2seq_model_core.check_tf_version_and_gpu()

    # set hyper parameters
    display_step = 300

    epochs = 10
    batch_size = 128

    rnn_size = 128
    num_layers = 3

    encoding_embedding_size = 200
    decoding_embedding_size = 200

    learning_rate = 0.001
    keep_probability = 0.5

    save_path = 'checkpoints/dev'
    (source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = seq2seq_model_core.load_preprocess()
    max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

    train_graph = tf.Graph()
    with train_graph.as_default():
        input_data, targets, target_sequence_length, max_target_sequence_length = seq2seq_model_core.enc_dec_model_inputs()
        lr, keep_prob = seq2seq_model_core.hyperparam_inputs()

        train_logits, inference_logits = seq2seq_model_core.seq2seq_model(tf.reverse(input_data, [-1]),
                                                            targets,
                                                            keep_prob,
                                                            batch_size,
                                                            target_sequence_length,
                                                            max_target_sequence_length,
                                                            len(source_vocab_to_int),
                                                            len(target_vocab_to_int),
                                                            encoding_embedding_size,
                                                            decoding_embedding_size,
                                                            rnn_size,
                                                            num_layers,
                                                            target_vocab_to_int)

        training_logits = tf.identity(train_logits.rnn_output, name='logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

        # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
        # - Returns a mask tensor representing the first N positions of each cell.
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function - weighted softmax cross entropy
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)



    # Split data to training and validation sets
    train_source = source_int_text[batch_size:]
    train_target = target_int_text[batch_size:]
    valid_source = source_int_text[:batch_size]
    valid_target = target_int_text[:batch_size]
    (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(seq2seq_model_core.get_batches(valid_source,
                                                                                                                 valid_target,
                                                                                                                 batch_size,
                                                                                                                 source_vocab_to_int['<PAD>'],
                                                                                                                 target_vocab_to_int['<PAD>']))
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(epochs):
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    seq2seq_model_core.get_batches(train_source, train_target, batch_size,
                                source_vocab_to_int['<PAD>'],
                                target_vocab_to_int['<PAD>'])):

                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: source_batch,
                     targets: target_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     keep_prob: keep_probability})


                if batch_i % display_step == 0 and batch_i > 0:
                    batch_train_logits = sess.run(
                        inference_logits,
                        {input_data: source_batch,
                         target_sequence_length: targets_lengths,
                         keep_prob: 1.0})

                    batch_valid_logits = sess.run(
                        inference_logits,
                        {input_data: valid_sources_batch,
                         target_sequence_length: valid_targets_lengths,
                         keep_prob: 1.0})

                    train_acc = seq2seq_model_core.get_accuracy(target_batch, batch_train_logits)
                    valid_acc = seq2seq_model_core.get_accuracy(valid_targets_batch, batch_valid_logits)

                    time = datetime.now().replace(microsecond=0)
                    time_str = time.strftime('%Y-%m-%d-%H-%M')
                    print('{} Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                          .format(time_str, epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print('Model Trained and Saved')

    # Save parameters for checkpoint
    seq2seq_model_core.save_params(save_path)