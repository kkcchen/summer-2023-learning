from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X, y = fetch_california_housing(return_X_y=True)
mod = KNeighborsRegressor().fit(X, y)


print(fetch_california_housing().DESCR)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor())
])


pipe.fit(X, y)


mod3 = GridSearchCV(estimator=pipe,
             param_grid={'model__n_neighbors': range(15, 20)},
             cv=3)
mod3.fit(X, y)


pred1 = mod.predict(X)
pred2 = pipe.predict(X)
pred3 = mod3.predict(X)


fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(pred1, y)
axs[0, 1].scatter(pred2, y)
axs[1, 0].scatter(pred3, y)

display(pd.DataFrame(mod3.cv_results_))

plt.show()
