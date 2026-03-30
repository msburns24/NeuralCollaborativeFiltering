# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.use_absl_handler()
absl.logging.set_verbosity(absl.logging.ERROR)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
from time import time

import numpy as np
from keras import initializers
from keras.layers import Concatenate, Dense, Embedding, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.regularizers import l2

from Dataset import Dataset
from evaluate import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run MLP.')
    parser.add_argument(
        '--path', nargs='?', default='Data/', help='Input data path.'
    )
    parser.add_argument(
        '--dataset', nargs='?', default='ml-1m', help='Choose a dataset.'
    )
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of epochs.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256, help='Batch size.'
    )
    parser.add_argument(
        '--layers', nargs='?', default='[64,32,16,8]',
        help=(
            'Size of each layer. Note that the first layer is the '
            'concatenation of user and item embeddings. So layers[0]/2 is '
            'the embedding size.'
        )
    )
    parser.add_argument(
        '--reg_layers', nargs='?', default='[0,0,0,0]',
        help='Regularization for each layer.'
    )
    parser.add_argument(
        '--num_neg', type=int, default=4,
        help='Number of negative instances to pair with a positive instance.'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate.'
    )
    parser.add_argument(
        '--learner', nargs='?', default='adam',
        help='Specify an optimizer: adagrad, adam, rmsprop, sgd'
    )
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='Show performance per X iterations'
    )
    parser.add_argument(
        '--out', type=int, default=1,
        help='Whether to save the trained model.'
    )
    return parser.parse_args()


def get_model(
    num_users: int,
    num_items: int,
    layers: list[int] = [20, 10],
    reg_layers: list[float] = [0, 0],
) -> Model:
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    mlp_embedding_user = Embedding(
        input_dim=num_users, output_dim=layers[0] // 2,
        name='user_embedding',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), input_length=1,
    )
    mlp_embedding_item = Embedding(
        input_dim=num_items, output_dim=layers[0] // 2,
        name='item_embedding',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), input_length=1,
    )

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(mlp_embedding_user(user_input))
    item_latent = Flatten()(mlp_embedding_item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    vector = Concatenate()([user_latent, item_latent])

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(
            layers[idx], kernel_regularizer=l2(reg_layers[idx]),
            activation='relu', name='layer%d' % idx,
        )
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(
        1, activation='sigmoid',
        kernel_initializer='lecun_uniform', name='prediction',
    )(vector)

    return Model(inputs=[user_input, item_input], outputs=prediction)


def get_train_instances(
    train,
    num_negatives: int,
) -> tuple[list[int], list[int], list[int]]:
    user_input: list[int] = []
    item_input: list[int] = []
    labels: list[int] = []
    for (u, i) in train.keys():
        # Positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # Negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1
    print('MLP arguments: %s' % args)
    model_out_file = 'Pretrain/%s_MLP_%s_%d.weights.h5' % (
        args.dataset, args.layers, time()
    )

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train = dataset.train_matrix
    test_ratings = dataset.test_ratings
    test_negatives = dataset.test_negatives
    num_users, num_items = dataset.num_users, dataset.num_items
    print(
        'Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d'
        % (time() - t1, num_users, num_items, train.nnz, len(test_ratings))
    )

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == 'adagrad':
        model.compile(
            optimizer=Adagrad(learning_rate=learning_rate),
            loss='binary_crossentropy',
        )
    elif learner.lower() == 'rmsprop':
        model.compile(
            optimizer=RMSprop(learning_rate=learning_rate),
            loss='binary_crossentropy',
        )
    elif learner.lower() == 'adam':
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
        )
    else:
        model.compile(
            optimizer=SGD(learning_rate=learning_rate),
            loss='binary_crossentropy',
        )

    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(
        model, test_ratings, test_negatives, topK, evaluation_threads
    )
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(
            train, num_negatives
        )
        # Training
        hist = model.fit(
            [np.array(user_input), np.array(item_input)],
            np.array(labels),
            batch_size=batch_size, epochs=1, verbose=0, shuffle=True,
        )
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(
                model, test_ratings, test_negatives, topK, evaluation_threads
            )
            hr, ndcg, loss = (
                np.array(hits).mean(),
                np.array(ndcgs).mean(),
                hist.history['loss'][0],
            )
            print(
                'Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, '
                'loss = %.4f [%.1f s]'
                % (epoch, t2 - t1, hr, ndcg, loss, time() - t2)
            )
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print(
        'End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. '
        % (best_iter, best_hr, best_ndcg)
    )
    if args.out > 0:
        print('The best MLP model is saved to %s' % model_out_file)
