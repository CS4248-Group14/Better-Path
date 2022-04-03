Code here is largely adapted from Zhou et al.'s original work from 
`https://github.com/YilunZhou/path-naturalness-prediction`.

Some changes are done to all `Python` files for it to be compatible in `Python3`, including
1. Change `print xx` to `print(xx)`.
2. Replace `xrange` with `range`.
3. Add `encoding='latin1'` for pickle loading.
4. Convert `zip` results to `list` before subscripting.

`heuristic.py` is created from scratch:
1. [DONE by Jiayu] Create multiple heuristic extractors and prepares cached heuristics.

`feature.py` is created from scratch:
1. [DONE by Wang Qian] Create L2 feature extractor

`dataset.py` is adapted from `code/science/dataset.py`.
Main changes:
1. [DONE by Tian Fang] Extract methods related to loading pre-generated files, and fix Zhou's mistake of not closing file descriptor.
2. [DONE by Wang Qian] New features for edges: vertex norm

`model.py` is adapted from `code/science/model.py`.
Main changes:
2. [DONE by Jiayu] Modify to avoid `RuntimeError` caused by inplace update (under CPU environment).
3. [DONE by Brendan] Two modified encoding methods (with dimension unification).
4. [DONE by Jiayu] More layers for predictor. 
   (Also commented a multi-layer model for `FeatureTransformer`, but not directly used because by experiments it 
   doesn't show great advantage. See [Experiment Results](../analysis/experiment_data.xlsx).)

`learn.py` is adapted from `code/science/learn.py`.
Main changes:
2. [DONE by Jiayu] Concat path heuristics in training.

