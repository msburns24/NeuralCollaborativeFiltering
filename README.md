# Neural Collaborative Filtering

This is an updated implementation of the paper:

> Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and
> Tat-Seng Chua (2017). [Neural Collaborative Filtering.][NCFPaper]
> In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

[NCFPaper]: http://dl.acm.org/citation.cfm?id=3052569

The paper introduces three collaborative filtering models for implicit
feedback: **Generalized Matrix Factorization (GMF)**,
**Multi-Layer Perceptron (MLP)**, and **Neural Matrix Factorization (NeuMF)**
— a fusion model that combines both. All three are trained with log loss
and negative sampling.

This repository is a fork of the [original authors' implementation][authorRepo],
updated to run on modern Python and TensorFlow/Keras.

[authorRepo]: https://github.com/hexiangnan/neural_collaborative_filtering

--------------------------------------------------------------------------------

## Changes from the Original

The original code targeted Keras 1.0.7 with a Theano backend, both of which are
long-deprecated. This fork modernizes the implementation while preserving the
architecture and training logic described in the paper.

**Framework migration (Keras 1 → Keras 3 / TF 2.x)**

- Replaced all legacy Keras 1 APIs with their modern equivalents
  (`Model`, `Embedding`, `Dense`, etc. from `keras` 3)
- Switched model serialization from `.h5` weight format to the current
  `.weights.h5` convention
- Replaced deprecated `fit_generator` and multi-output patterns with
  the standard `model.fit` API
- Added TF warning suppression for cleaner training output

**Performance: batched evaluation**

- The original `evaluate.py` called `model.predict` once per user,
  which was extremely slow on large datasets. Replaced with a single
  batched prediction over all users, giving a significant speedup at
  evaluation time.

**Bug fix: embedding size mismatch**

- `Dataset.py` now extends `num_users` and `num_items` to cover IDs
  appearing in the test set and negatives that may be absent from
  training data. The original code could produce out-of-bounds
  embedding lookups on sampled datasets.

**CLI modernization**

- Replaced `argparse` with
  [`typed-argument-parser`](https://github.com/swansonk14/typed-argument-parser)
  (`tap`), giving type-annotated, self-documenting argument
  definitions.

**New: dataset sampling utility**

- Added `sample_dataset.py` to create small, self-contained dataset
  subsets for fast iteration and testing. Handles dense ID remapping
  and resamples negatives from the training item pool to ensure every
  item scored at eval time has a trained embedding.

--------------------------------------------------------------------------------

## Environment

- Python 3.11+
- TensorFlow 2.x / Keras 3
- See `requirements.txt` for full dependencies

```bash
pip install -r requirements.txt
```

--------------------------------------------------------------------------------

## Quickstart

### Run GMF

```bash
python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 \
  --num_factors 8 --regs '[0,0]' --num_neg 4 --lr 0.001 \
  --learner adam --verbose 1 --out
```

### Run MLP

```bash
python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 \
  --layers '[64,32,16,8]' --reg_layers '[0,0,0,0]' --num_neg 4 \
  --lr 0.001 --learner adam --verbose 1 --out
```

### Run NeuMF (without pre-training)

```bash
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 \
  --num_factors 8 --layers '[64,32,16,8]' --reg_mf 0 \
  --reg_layers '[0,0,0,0]' --num_neg 4 --lr 0.001 --learner adam \
  --verbose 1 --out
```

### Run NeuMF (with pre-training)

```bash
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 \
  --num_factors 8 --layers '[64,32,16,8]' --num_neg 4 --lr 0.001 \
  --learner adam --verbose 1 --out \
  --mf_pretrain Pretrain/ml-1m_GMF_8_<timestamp>.weights.h5 \
  --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_<timestamp>.weights.h5
```

> **Note on pre-training:** For small embedding dimensions, NeuMF
> without pre-training often matches or beats GMF and MLP individually.
> Pre-training tends to help more with larger embedding sizes, and may
> require tuning regularization for the GMF and MLP components.

> **Shell note:** Array arguments like `--layers '[64,32,16,8]'` must
> be quoted to prevent shell expansion. Single quotes work in bash and
> zsh; on Windows CMD, use double quotes.

--------------------------------------------------------------------------------

## Generating a Small Sample Dataset

For quick testing without running on the full dataset:

```bash
python sample_dataset.py --dataset ml-1m --num_users 500
```

This creates a self-contained subset at `Data/sample-ml-1m/` with remapped
user/item IDs and resampled negatives. You can then run any of the models
against it:

```bash
python NeuMF.py --dataset sample-ml-1m --epochs 5 --batch_size 256 \
  --num_factors 8 --layers '[64,32,16,8]' --reg_mf 0 \
  --reg_layers '[0,0,0,0]' --num_neg 4 --lr 0.001 --learner adam \
  --verbose 1
```

--------------------------------------------------------------------------------

## Dataset Format

Two datasets are included: MovieLens 1M (`ml-1m`) and Pinterest
(`pinterest-20`), located in `Data/`.

| File | Description |
|---|---|
| `<dataset>.train.rating` | Training interactions: `userID\titemID\trating\ttimestamp` |
| `<dataset>.test.rating` | One held-out positive per user: same format |
| `<dataset>.test.negative` | 99 negative samples per user: `(userID,itemID)\tneg1\tneg2\t...` |

--------------------------------------------------------------------------------

## Acknowledgements

Original implementation by [Dr. Xiangnan He](http://www.comp.nus.edu.sg/~xiangnan/)
and co-authors. Please cite the WWW '17 paper if you use this code in
your work.
