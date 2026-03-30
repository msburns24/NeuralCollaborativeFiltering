'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import scipy.sparse as sp


class Dataset:
    def __init__(self, path: str) -> None:
        self.train_matrix = self._load_rating_file_as_matrix(
            path + '.train.rating'
        )
        self.test_ratings = self._load_rating_file_as_list(
            path + '.test.rating'
        )
        self.test_negatives = self._load_negative_file(
            path + '.test.negative'
        )
        assert len(self.test_ratings) == len(self.test_negatives)

        self.num_users, self.num_items = self.train_matrix.get_shape()

        # The train matrix is sized to the max IDs seen in training only.
        # Test ratings and negatives may reference items (or users) not present
        # in training (common in sampled datasets). Extend the counts so that
        # embedding matrices cover every ID that will be looked up at eval time.
        if self.test_ratings:
            max_test_user = max(r[0] for r in self.test_ratings)
            max_test_item = max(r[1] for r in self.test_ratings)
            self.num_users = max(self.num_users, max_test_user + 1)
            self.num_items = max(self.num_items, max_test_item + 1)
        if self.test_negatives:
            max_neg_item = max(
                max(negs) for negs in self.test_negatives if negs
            )
            self.num_items = max(self.num_items, max_neg_item + 1)

    def _load_rating_file_as_list(self, filename: str) -> list[list[int]]:
        rating_list: list[list[int]] = []
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                arr = line.split('\t')
                rating_list.append([int(arr[0]), int(arr[1])])
        return rating_list

    def _load_negative_file(self, filename: str) -> list[list[int]]:
        negative_list: list[list[int]] = []
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                arr = line.split('\t')
                negative_list.append([int(x) for x in arr[1:]])
        return negative_list

    def _load_rating_file_as_matrix(self, filename: str) -> sp.dok_matrix:
        '''Read .rating file and return dok matrix.'''
        rows: list[tuple[int, int, float]] = []
        num_users, num_items = 0, 0
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                arr = line.split('\t')
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                num_users = max(num_users, user)
                num_items = max(num_items, item)
                rows.append((user, item, rating))
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        for user, item, rating in rows:
            if rating > 0:
                mat[user, item] = 1.0
        return mat
