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
from keras.layers import Dense, Embedding, Flatten, Input, Multiply
from keras.models import Model
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.regularizers import l2

from Dataset import Dataset
from evaluate import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run GMF.')
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
        '--num_factors', type=int, default=8, help='Embedding size.'
    )
    parser.add_argument(
        '--regs', nargs='?', default='[0,0]',
        help='Regularization for user and item embeddings.'
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
    latent_dim: int,
    regs: list[float] = [0, 0],
) -> Model:
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    mf_embedding_user = Embedding(
        input_dim=num_users, output_dim=latent_dim, name='user_embedding',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),  # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(regs[0]),
    )
    mf_embedding_item = Embedding(
        input_dim=num_items, output_dim=latent_dim, name='item_embedding',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),  # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(regs[1]),
    )

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(mf_embedding_user(user_input))
    item_latent = Flatten()(mf_embedding_item(item_input))

    # Element-wise product of user and item embeddings
    predict_vector = Multiply()([user_latent, item_latent])

    # Final prediction layer
    prediction = Dense(
        1, activation='sigmoid',
        kernel_initializer='lecun_uniform', name='prediction',
    )(predict_vector)

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
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1
    print('GMF arguments: %s' % args)
    model_out_file = 'Pretrain/%s_GMF_%d_%d.weights.h5' % (
        args.dataset, num_factors, time()
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
    model = get_model(num_users, num_items, num_factors, regs)
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
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))

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
        print('The best GMF model is saved to %s' % model_out_file)
