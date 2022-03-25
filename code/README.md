Code here is largely adapted from Zhou et al.'s original work from 
`https://github.com/YilunZhou/path-naturalness-prediction`.

`dataset.py` is adapted from `code/science/dataset.py`.
Changes:
1. `Python` version compatibility issue, change `print xx` to `print(xx)`, `xrange` to `range`.
1. Add two parameters for feature and data paths respectively in class constructor (for implementation flexibility).
1. [TODO @Wang Qian] New features for edges.
1. [TODO @Jiayu] Path heuristics util.

`model.py` is adapted from `code/science/model.py`.
Changes:
1. `Python` version compatibility issue, change `xrange` to `range`.
1. [TODO @Tian Fang, @Brendan] Two modified encoding methods (with dimension unification).

`learn.py` is adapted from `code/science/learn.py`.
Changes:
1. `Python` version compatibility issue, change `print xx` to `print(xx)`, `xrange` to `range`.
1. Wrap code in main function.
1. Update paths according to configuration in our project.
