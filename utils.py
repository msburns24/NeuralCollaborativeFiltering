import numpy as np
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import Optimizer
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from scipy.sparse import dok_matrix


def get_train_instances(
    train: dok_matrix,
    num_items: int,
    num_negatives: int,
) -> tuple[list[int], list[int], list[int]]:
    user_input: list[int] = []
    item_input: list[int] = []
    labels: list[int] = []

    for (u, i) in train.keys(): # pyright: ignore[reportGeneralTypeIssues]
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


def get_optimizer_by_name(name: str, *args, **kwargs) -> Optimizer:
    choices = {'adagrad': Adagrad, 'rmsprop': RMSprop, 'adam': Adam}
    OptimizerClass = choices.get(name.strip().lower(), SGD)
    return OptimizerClass(*args, **kwargs)
