'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import numpy as np

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    num_users = len(testRatings)
    gt_items = []
    users_list = []
    items_list = []
    counts = []

    for idx in range(num_users):
        u = testRatings[idx][0]
        gt_item = testRatings[idx][1]
        items = testNegatives[idx] + [gt_item]
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

    hits, ndcgs = [], []
    offset = 0
    for idx in range(num_users):
        count = counts[idx]
        preds = all_predictions[offset:offset + count]
        offset += count

        items = testNegatives[idx] + [gt_items[idx]]
        map_item_score = dict(zip(items, preds))
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hits.append(getHitRatio(ranklist, gt_items[idx]))
        ndcgs.append(getNDCG(ranklist, gt_items[idx]))

    return (hits, ndcgs)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
