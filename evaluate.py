import heapq
import math

import numpy as np
from keras.models import Model


def evaluate_model(
    model: Model,
    test_ratings: list[list[int]],
    test_negatives: list[list[int]],
    k: int,
) -> tuple[list[float], list[float]]:
    '''Evaluate the performance (Hit Ratio, NDCG) of top-K recommendation.'''
    num_users = len(test_ratings)
    gt_items: list[int] = []
    users_list: list[np.ndarray] = []
    items_list: list[np.ndarray] = []
    counts: list[int] = []

    for idx in range(num_users):
        u = test_ratings[idx][0]
        gt_item = test_ratings[idx][1]
        items = test_negatives[idx] + [gt_item]
        count = len(items)
        users_list.append(np.full(count, u, dtype='int32'))
        items_list.append(np.array(items, dtype='int32'))
        gt_items.append(gt_item)
        counts.append(count)

    all_users = np.concatenate(users_list)
    all_items = np.concatenate(items_list)

    # Single batched prediction instead of one call per user
    all_predictions = model.predict(
        [all_users, all_items], batch_size=1024, verbose=0
    ).flatten()

    hits: list[float] = []
    ndcgs: list[float] = []
    offset = 0
    for idx in range(num_users):
        count = counts[idx]
        preds = all_predictions[offset:offset + count]
        offset += count

        items = test_negatives[idx] + [gt_items[idx]]
        map_item_score = dict(zip(items, preds))
        ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hits.append(get_hit_ratio(ranklist, gt_items[idx]))
        ndcgs.append(get_ndcg(ranklist, gt_items[idx]))

    return (hits, ndcgs)


def get_hit_ratio(ranklist: list[int], gt_item: int) -> float:
    for item in ranklist:
        if item == gt_item:
            return 1.0
    return 0.0


def get_ndcg(ranklist: list[int], gt_item: int) -> float:
    for i, item in enumerate(ranklist):
        if item == gt_item:
            return math.log(2) / math.log(i + 2)
    return 0.0
