"""Бустинг и лес на погоде новой площадки (для таблицы сравнения моделей)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
X = base_table()
evaluate(X, FEATURES, name="Линейная")
evaluate(X, FEATURES, lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1., random_state=0), name="Бустинг")
evaluate(X, FEATURES, lambda: RandomForestRegressor(200, min_samples_leaf=10, n_jobs=-1, random_state=0), name="Случайный лес")
evaluate(X, PROCESS + WINDOW, name="Без погоды")
