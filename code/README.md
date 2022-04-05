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
2. [DONE by Tian Fang] Create L1 feature extractor

`experiment.py` is created from scratch:
1. [DONE by Tian Fang] Integrate everyone's work (and make necessary changes in corresponding files).
2. [DONE by Tian Fang] Create multiple functions to perform ablation study on features, heuristics, encoder, and multilayer
3. [DONE by Tian Fang] Speed up ablation study via `multiprocessing`

`dataset.py` is adapted from `code/science/dataset.py`.
Main changes:
1. [DONE by Tian Fang] Extract methods related to loading pre-generated files, and fix Zhou's mistake of not closing file descriptor.
2. [DONE by Wang Qian] New features for edges: vertex norm

`model.py` is adapted from `code/science/model.py`.
Main changes:
1. [DONE by Jiayu] Modify to avoid `RuntimeError` caused by inplace update (under CPU environment).
2. [DONE by Brendan] Two modified encoding methods (with dimension unification).
3. [DONE by Jiayu] More layers for predictor.
4. [DONE by Tian Fang] Modify Predictor to allow turning multilayer on and off.
5. [DONE by Tian Fang] Create Encoder using Transformer's Encoder with a linear layer on top.

`learn.py` is adapted from `code/science/learn.py`.
Main changes:
1. [DONE by Jiayu] Concat path heuristics in training.
2. [DONE by Tian Fang] Added Enum class VertexEdgeEncoderType and add various input parameters to the `train()` method to facilitate ablation study
