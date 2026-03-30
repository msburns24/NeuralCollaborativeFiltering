# Suppress TF Warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import ast
from time import time

import numpy as np
from keras import initializers
from keras.layers import Concatenate, Dense, Embedding, Flatten, Input
from keras.models import Model
from keras.regularizers import l2

from cli import MLPArgs
from Dataset import Dataset
from evaluate import evaluate_model
from utils import get_optimizer_by_name
from utils import get_train_instances


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
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01), # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(reg_layers[0]),
    )
    mlp_embedding_item = Embedding(
        input_dim=num_items, output_dim=layers[0] // 2,
        name='item_embedding',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01), # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(reg_layers[0]),
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


if __name__ == '__main__':
    args = MLPArgs().parse_args()
    layers = ast.literal_eval(args.layers)
    reg_layers = ast.literal_eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1
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
    optimizer = get_optimizer_by_name(learner, learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

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
            train, num_items, num_negatives,
        )
        # Training
        hist = model.fit(
            [np.array(user_input), np.array(item_input)],
            np.array(labels),
            batch_size=batch_size, epochs=1, shuffle=True,
            verbose=0, # pyright: ignore[reportArgumentType]
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
