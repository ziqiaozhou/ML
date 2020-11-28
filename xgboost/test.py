import numpy as np

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn_gbmi import *
DATA_COUNT = 10000

RANDOM_SEED = 137

TRAIN_FRACTION = 0.9
np.random.seed(RANDOM_SEED)
xs = pd.DataFrame(np.random.random_integers(low=0, high=1,size = (DATA_COUNT, 3)))
xs.columns = ['x0', 'x1', 'x2']
y = pd.DataFrame((xs.x0|xs.x1) & xs.x2)
y.columns = ['y']
train_ilocs = range(int(TRAIN_FRACTION*DATA_COUNT))

test_ilocs = range(int(TRAIN_FRACTION*DATA_COUNT), DATA_COUNT)
gbr_1 = GradientBoostingRegressor(random_state = RANDOM_SEED)
gbr_1.fit(xs.iloc[train_ilocs], y.y.iloc[train_ilocs])
gbr_1.score(xs.iloc[test_ilocs], y.y.iloc[test_ilocs])
