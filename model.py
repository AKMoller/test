from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

class LassoEstimator(Lasso):
    """
        This class extends the Lasso class from sklearn, and inherits all its methods.
        The score function is overwritten with a mean squared error.
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super().__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)

    def score(self, X, y):
        mse = mean_squared_error(X, y)
        return mse
