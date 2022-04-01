Code here is largely adapted from Zhou et al.'s original work from 
`https://github.com/YilunZhou/path-naturalness-prediction`.

Some changes are done to all `Python` files for it to be compatible in `Python3`, including
1. Change `print xx` to `print(xx)`.
1. Replace `xrange` with `range`.
1. Add `encoding='latin1'` for pickle loading.
1. Convert `zip` results to `list` before subscripting.

`dataset.py` is adapted from `code/science/dataset.py`.
Changes:
1. [DONE by Jiayu] Add two parameters for feature and data paths respectively in class constructor (for implementation flexibility).
1. [TODO @Wang Qian] New features for edges.
1. [DONE by Jiayu] Prepare cached heuristics.
1. [TODO @Jiayu] Path heuristics util.

`model.py` is adapted from `code/science/model.py`.
Changes:
1. [DONE by Jiayu] Modify to avoid `RuntimeError` caused by inplace update (under CPU environment).
1. [TODO @Tian Fang, @Brendan] Two modified encoding methods (with dimension unification).
1. [DONE by Jiayu] More layers for predictor. 
   (Also commented a multi-layer model for `FeatureTransformer`, but not directly used because by experiments it 
   doesn't show great advantage. See [Experiment Results](../analysis/experiment_data.xlsx).)

`learn.py` is adapted from `code/science/learn.py`.
Changes:
1. [DONE by Jiayu] Wrap code in main function.
1. [DONE by Jiayu] Update paths according to configuration in our project.
1. [DONE by Jiayu] Add more hyperparameters to `train` function.
1. [TODO @Jiayu] Concat path heuristics in training.

