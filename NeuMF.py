# Suppress TensorFlow warnings
import os
import absl.logging
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.use_absl_handler()
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import ast
import argparse
from time import time

import numpy as np
from keras import initializers
from keras.layers import Concatenate, Dense, Embedding, Flatten, Input, Multiply
from keras.models import Model
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.regularizers import l2

import GMF
import MLP
from Dataset import Dataset
from evaluate import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run NeuMF.')
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
        '--num_factors', type=int, default=8,
        help='Embedding size of MF model.'
    )
    parser.add_argument(
        '--layers', nargs='?', default='[64,32,16,8]',
        help=(
            'MLP layers. Note that the first layer is the concatenation of '
            'user and item embeddings. So layers[0]/2 is the embedding size.'
        )
    )
    parser.add_argument(
        '--reg_mf', type=float, default=0,
        help='Regularization for MF embeddings.'
    )
    parser.add_argument(
        '--reg_layers', nargs='?', default='[0,0,0,0]',
        help=(
            'Regularization for each MLP layer. reg_layers[0] is the '
            'regularization for embeddings.'
        )
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
    parser.add_argument(
        '--mf_pretrain', nargs='?', default='',
        help=(
            'Specify the pretrain model file for MF part. '
            'If empty, no pretrain will be used.'
        )
    )
    parser.add_argument(
        '--mlp_pretrain', nargs='?', default='',
        help=(
            'Specify the pretrain model file for MLP part. '
            'If empty, no pretrain will be used.'
        )
    )
    return parser.parse_args()


def get_model(
    num_users: int,
    num_items: int,
    mf_dim: int = 10,
    layers: list[int] = [10],
    reg_layers: list[float] = [0],
    reg_mf: float = 0,
) -> Model:
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layers
    mf_embedding_user = Embedding(
        input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_mf), input_length=1,
    )
    mf_embedding_item = Embedding(
        input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_mf), input_length=1,
    )
    mlp_embedding_user = Embedding(
        input_dim=num_users, output_dim=layers[0] // 2,
        name='mlp_embedding_user',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), input_length=1,
    )
    mlp_embedding_item = Embedding(
        input_dim=num_items, output_dim=layers[0] // 2,
        name='mlp_embedding_item',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), input_length=1,
    )

    # MF part
    mf_user_latent = Flatten()(mf_embedding_user(user_input))
    mf_item_latent = Flatten()(mf_embedding_item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = Flatten()(mlp_embedding_user(user_input))
    mlp_item_latent = Flatten()(mlp_embedding_item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(
            layers[idx], kernel_regularizer=l2(reg_layers[idx]),
            activation='relu', name='layer%d' % idx,
        )
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(
        1, activation='sigmoid',
        kernel_initializer='lecun_uniform', name='prediction',
    )(predict_vector)

    return Model(inputs=[user_input, item_input], outputs=prediction)


def load_pretrain_model(
    model: Model,
    gmf_model: Model,
    mlp_model: Model,
    num_layers: int,
) -> Model:
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate(
        (gmf_prediction[0], mlp_prediction[0]), axis=0
    )
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model


def get_train_instances(
    train,
    num_items: int,
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
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = ast.literal_eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = ast.literal_eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1
    print('NeuMF arguments: %s' % args)
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.weights.h5' % (
        args.dataset, mf_dim, args.layers, time()
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
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
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

    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMF.get_model(num_users, num_items, mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print(
            'Load pretrained GMF (%s) and MLP (%s) models done. '
            % (mf_pretrain, mlp_pretrain)
        )

    # Init performance
    (hits, ndcgs) = evaluate_model(
        model, test_ratings, test_negatives, topK, evaluation_threads
    )
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)

    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(
            train, num_items, num_negatives
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
        print('The best NeuMF model is saved to %s' % model_out_file)
