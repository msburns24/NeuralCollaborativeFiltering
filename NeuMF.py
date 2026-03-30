# Suppress TF Warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import ast
from time import time

import numpy as np
from keras import initializers
from keras.layers import Concatenate, Dense, Embedding, Flatten, Input, Multiply
from keras.models import Model
from keras.regularizers import l2

import GMF
import MLP
from cli import NeuMFArgs
from Dataset import Dataset
from evaluate import evaluate_model
from utils import get_optimizer_by_name
from utils import get_train_instances


def get_model(
    num_users: int,
    num_items: int,
    mf_dim: int = 10,
    layers: list[int] | None = None,
    reg_layers: list[float] | None = None,
    reg_mf: float = 0,
) -> Model:
    if layers is None:
        layers = [10]
    if reg_layers is None:
        reg_layers = [0]
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layers
    mf_embedding_user = Embedding(
        input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01), # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(reg_mf),
    )
    mf_embedding_item = Embedding(
        input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01), # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(reg_mf),
    )
    mlp_embedding_user = Embedding(
        input_dim=num_users, output_dim=layers[0] // 2,
        name='mlp_embedding_user',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01), # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(reg_layers[0]),
    )
    mlp_embedding_item = Embedding(
        input_dim=num_items, output_dim=layers[0] // 2,
        name='mlp_embedding_item',
        embeddings_initializer=initializers.TruncatedNormal(stddev=0.01), # pyright: ignore[reportArgumentType]
        embeddings_regularizer=l2(reg_layers[0]),
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


if __name__ == '__main__':
    args = NeuMFArgs().parse_args()
    layers = ast.literal_eval(args.layers)
    reg_layers = ast.literal_eval(args.reg_layers)
    topK = 10
    print('NeuMF arguments: %s' % args)
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.weights.h5' % (
        args.dataset, args.num_factors, args.layers, time()
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
    model = get_model(
        num_users, num_items, args.num_factors, layers, reg_layers, args.reg_mf
    )
    optimizer = get_optimizer_by_name(args.learner, learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    # Load pretrain model
    if args.mf_pretrain != '' and args.mlp_pretrain != '':
        gmf_model = GMF.get_model(num_users, num_items, args.num_factors)
        gmf_model.load_weights(args.mf_pretrain)
        mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights(args.mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print(
            'Load pretrained GMF (%s) and MLP (%s) models done. '
            % (args.mf_pretrain, args.mlp_pretrain)
        )

    # Init performance
    (hits, ndcgs) = evaluate_model(
        model, test_ratings, test_negatives, topK
    )
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out:
        model.save_weights(model_out_file, overwrite=True)

    # Training model
    for epoch in range(args.epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(
            train, num_items, args.num_neg
        )
        # Training
        hist = model.fit(
            [np.array(user_input), np.array(item_input)],
            np.array(labels),
            batch_size=args.batch_size, epochs=1, shuffle=True,
            verbose=0, # pyright: ignore[reportArgumentType]
        )
        t2 = time()

        # Evaluation
        if epoch % args.verbose == 0:
            (hits, ndcgs) = evaluate_model(
                model, test_ratings, test_negatives, topK
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
                if args.out:
                    model.save_weights(model_out_file, overwrite=True)

    print(
        'End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. '
        % (best_iter, best_hr, best_ndcg)
    )
    if args.out:
        print('The best NeuMF model is saved to %s' % model_out_file)
