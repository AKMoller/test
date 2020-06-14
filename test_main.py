from model import LassoEstimator

def test_LassoEstimator_score():

    a = [1, 2, 3, 4]
    b = [2, 3, 4, 5]

    lasso = LassoEstimator()

    assert lasso.score(a, b) == 1
