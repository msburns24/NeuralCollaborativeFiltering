# Suppress TF Warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import ast
from time import time

import numpy as np
from keras import initializers
from keras.layers import Dense, Embedding, Flatten, Input, Multiply
from keras.models import Model
from keras.regularizers import l2

from cli import GMFArgs
from Dataset import Dataset
from evaluate import evaluate_model
from utils import get_optimizer_by_name
from utils import get_train_instances


def get_model(
    num_users: int,
    num_items: int,
    latent_dim: int,
    regs: list[float] | None = None,
) -> Model:
    if regs is None:
        regs = [0, 0]
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


if __name__ == '__main__':
    args = GMFArgs().parse_args()
    num_factors = args.num_factors
    regs = ast.literal_eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    topK = 10
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
    optimizer = get_optimizer_by_name(learner, learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(
        model, test_ratings, test_negatives, topK
    )
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))

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
        print('The best GMF model is saved to %s' % model_out_file)
