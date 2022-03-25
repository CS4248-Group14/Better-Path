Code here is largely adapted from Zhou et al.'s original work from 
`https://github.com/YilunZhou/path-naturalness-prediction`.

`dataset.py` is adapted from `code/science/dataset.py`.
Changes:
1. `Python` version compatibility issue, change `print xx` to `print(xx)`, `xrange` to `range`.
1. Add two parameters for feature and data paths respectively in class constructor (for implementation flexibility).
