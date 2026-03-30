'''
Create a small sample of a dataset for quick testing.

Reads the first --num_users users from a dataset and writes a new dataset to
Data/sample-<dataset>/ with:
  - item IDs remapped to a dense 0-indexed space covering only items that appear
    in train or test.rating (no negatives-only items)
  - user IDs remapped to dense 0-indexed space
  - 99 negatives per user resampled from the training item pool
    (guarantees every item in the negatives has a trained embedding)

Usage:
    python sample_dataset.py [--dataset ml-1m] [--num_users 500] [--path Data/]
    python sample_dataset.py --dataset pinterest-20 --num_users 200
'''

import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a small dataset sample for testing.'
    )
    parser.add_argument('--path', default='Data/', help='Data directory.')
    parser.add_argument('--dataset', default='ml-1m', help='Dataset name.')
    parser.add_argument(
        '--num_users', type=int, default=500,
        help='Number of users to include in the sample.'
    )
    parser.add_argument(
        '--num_neg', type=int, default=99,
        help='Number of negative items to sample per test user (default: 99).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for negative sampling.'
    )
    return parser.parse_args()


def read_lines(path: str) -> list[str]:
    with open(path, 'r') as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    path = args.path
    dataset = args.dataset
    num_users = args.num_users
    num_neg = args.num_neg
    rng = np.random.default_rng(args.seed)

    prefix = os.path.join(path, dataset)

    train_path = prefix + '.train.rating'
    test_r_path = prefix + '.test.rating'
    test_neg_path = prefix + '.test.negative'

    for p in (train_path, test_r_path, test_neg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'Expected data file not found: {p}')

    # --- Load test files (one row per user, row-index-aligned) ---
    test_rating_rows = read_lines(test_r_path)
    test_negative_rows = read_lines(test_neg_path)

    if len(test_rating_rows) != len(test_negative_rows):
        raise ValueError(
            f'test.rating has {len(test_rating_rows)} rows but '
            f'test.negative has {len(test_negative_rows)} rows '
            f'\u2014 files are misaligned.'
        )

    num_users = min(num_users, len(test_rating_rows))

    # --- Determine the selected user IDs (first num_users rows of test.rating) ---
    # test.rating rows are ordered by user_id; collect them in order for remapping.
    selected_user_ids_ordered = []
    for row in test_rating_rows[:num_users]:
        uid = int(row.split('\t')[0])
        selected_user_ids_ordered.append(uid)

    selected_user_set = set(selected_user_ids_ordered)
    user_map = {old: new for new, old in enumerate(selected_user_ids_ordered)}

    # --- Load and filter train.rating ---
    train_rows_by_user: dict[int, list[list[str]]] = {}
    with open(train_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            uid = int(parts[0])
            if uid in selected_user_set:
                train_rows_by_user.setdefault(uid, []).append(parts)

    # --- Collect item IDs from train and test.rating only (no negatives) ---
    # Items that appear solely in the original negatives pool are excluded:
    # they would never receive a gradient update, making the evaluation less
    # meaningful for a small sample.
    all_items: set[int] = set()

    for rows in train_rows_by_user.values():
        for parts in rows:
            all_items.add(int(parts[1]))

    for row in test_rating_rows[:num_users]:
        parts = row.split('\t')
        all_items.add(int(parts[1]))

    # Build dense item remapping (sorted so it's reproducible)
    item_map = {old: new for new, old in enumerate(sorted(all_items))}

    num_new_users = len(user_map)
    num_new_items = len(item_map)
    train_count = sum(len(v) for v in train_rows_by_user.values())
    # The pool of remapped item IDs available for negative sampling
    all_new_item_ids = np.array(sorted(item_map.values()), dtype=np.int32)

    print(f'Dataset:       {dataset}')
    print(
        f'Users:         {num_new_users}  (original IDs: '
        f'{selected_user_ids_ordered[0]}\u2013{selected_user_ids_ordered[-1]})'
    )
    print(f'Items:         {num_new_items} (remapped to 0\u2013{num_new_items - 1})')
    print(f'Train rows:    {train_count}')
    print(f'Test rows:     {num_new_users}')

    # --- Set up output directory ---
    out_dataset = f'sample-{dataset}'
    out_dir = os.path.join(path, out_dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, out_dataset)

    # --- Write train.rating ---
    with open(out_prefix + '.train.rating', 'w') as f:
        for old_uid in selected_user_ids_ordered:
            new_uid = user_map[old_uid]
            for parts in train_rows_by_user.get(old_uid, []):
                new_iid = item_map[int(parts[1])]
                # Preserve rating and timestamp columns as-is
                f.write(f'{new_uid}\t{new_iid}\t{parts[2]}\t{parts[3]}\n')

    # --- Write test.rating ---
    with open(out_prefix + '.test.rating', 'w') as f:
        for row in test_rating_rows[:num_users]:
            parts = row.split('\t')
            new_uid = user_map[int(parts[0])]
            new_iid = item_map[int(parts[1])]
            f.write(f'{new_uid}\t{new_iid}\t{parts[2]}\t{parts[3]}\n')

    # --- Write test.negative ---
    # Negatives are resampled from the training item pool so that every item
    # the model is asked to score has a trained embedding.
    # Format: (new_uid,new_pos_iid)\tneg1\tneg2\t...
    with open(out_prefix + '.test.negative', 'w') as f:
        for row in test_rating_rows[:num_users]:
            parts = row.split('\t')
            old_uid = int(parts[0])
            old_pos_iid = int(parts[1])
            new_uid = user_map[old_uid]
            new_pos_iid = item_map[old_pos_iid]

            # Items this user interacted with in train + their test item
            user_pos_new = {
                item_map[int(p[1])]
                for p in train_rows_by_user.get(old_uid, [])
            }
            user_pos_new.add(new_pos_iid)

            # Sample without replacement from items the user hasn't interacted with
            candidate_mask = np.isin(
                all_new_item_ids, list(user_pos_new), invert=True
            )
            candidates = all_new_item_ids[candidate_mask]
            replace = len(candidates) < num_neg
            sampled = rng.choice(candidates, size=num_neg, replace=replace)

            f.write(
                f'({new_uid},{new_pos_iid})\t'
                + '\t'.join(map(str, sampled)) + '\n'
            )

    print(f'\nSample written to: {out_dir}/')


if __name__ == '__main__':
    main()
