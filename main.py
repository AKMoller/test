import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.neural_network import MLPRegressor
from model import LassoEstimator

# Read the data, and format to float.
features = pd.read_pickle("./features.pkl").astype(float)
target = pd.read_pickle("./target.pkl").astype(float)[0]

# Create new lagged/first difference variables
features['diff0'] = features[0].diff()
features['diff1'] = features[1].diff()
features['lag2'] = features[2].shift(1)

# Remove nulls.
null_idx = features.isnull()
null_idx_all = np.sum(null_idx, axis=1)
features = features.loc[null_idx_all == 0]
target = target.loc[null_idx_all == 0]

# Plot the data (target, features and correlation)
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(target)
axs[0, 1].plot(features[0][1:1500], target[1:1500], 'ro')
axs[1, 0].plot(features[0][1:1500])
axs[1, 0].plot(target[1:1500])
axs[1, 1].plot(features[0][2000:4500])
axs[1, 1].plot(features[1][2000:4500])
axs[1, 1].plot(features[2][2000:4500])

# Split the data into train, validation and test datasets (60/20/20).
nrows = features.shape[0]
index = np.zeros(nrows)
index[int(0.6*nrows):] = 1
index[int(0.8*nrows):] = 2

features_train = features.loc[index == 0]
features_validation = features.loc[index == 1]
features_test = features.loc[index == 2]

target_train = target.loc[index == 0]
target_validation = target.loc[index == 1]
target_test = target.loc[index == 2]

# Check if one of the FC models is significantly better.
forecast_mse = [((target-features[0])**2).mean(),
                ((target-features[1])**2).mean(),
                ((target-features[2])**2).mean()]
print(forecast_mse)

################ LASSO ################

# Parameter grid to go through.
param_grid_lasso = {'alpha': [0.001, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                              0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1]}

# Construct (random) combinations of the parameters (in this case just all of them)
N_ITER = len(param_grid_lasso['alpha'])
param_sampler = ParameterSampler(param_grid_lasso, n_iter=N_ITER)

# Preallocate space for the results
lasso_validation_results = pd.DataFrame(np.zeros((N_ITER, len(param_grid_lasso)+2)),
                                        columns=['alpha', 'coeff', 'score'])

# Loop over the different parameters.
for idx, param in enumerate(param_sampler):
    lasso_obj = LassoEstimator(**param)
    lasso_obj.fit(features_train, target_train)
    predict = lasso_obj.predict(features_validation)
    score = lasso_obj.score(target_validation, predict)
    lasso_validation_results.iloc[idx, :] = [param['alpha'],
                                             lasso_obj.coef_.astype('object'),
                                             score]

print(lasso_validation_results)

# Get parameters of the best performing model
idx_lasso_model = np.argmin(lasso_validation_results['score'])
lasso_model = lasso_validation_results['alpha'][idx_lasso_model]

################ NN ################

# Try with a model with no assumptions on the underlying structure
param_grid_nn = {'hidden_layer_sizes': [(2, 1), (2,)],
                 'activation': ['identity', 'logistic', 'tanh', 'relu'],
                 'learning_rate_init': [0.0005, 0.001, 0.002]}

N_ITER_NN = 10
param_sampler_nn = ParameterSampler(param_grid_nn, n_iter=N_ITER_NN, random_state=0)

cols = sorted(list(param_grid_nn.keys()))
nn_validation_results = pd.DataFrame(np.zeros((N_ITER_NN, len(cols)+1)), columns=cols+['score'])

for idx, param in enumerate(param_sampler_nn):
    NN = MLPRegressor(**param)

    NN.fit(features_train, target_train)
    predict = NN.predict(features_validation)
    score = mean_squared_error(target_validation, predict)

    #score_df.iloc[idx, :] = [param[x] for x in cols]+list(score)
    nn_validation_results.iloc[idx, -1] = score
    nn_validation_results.iloc[idx, 0] = param['activation']
    nn_validation_results.iloc[idx, 1] = str(param['hidden_layer_sizes'])
    nn_validation_results.iloc[idx, 2] = param['learning_rate_init']

print(nn_validation_results)

################ Results ################

# Run the chosen model on the test data. Compare with the 3 FC alone.
lasso_validation = LassoEstimator(alpha=lasso_model)
lasso_validation.fit(features_test, target_test)
lasso_test_predict = lasso_validation.predict(features_test)

nn_validation = MLPRegressor(activation='tanh', hidden_layer_sizes=(2,), learning_rate_init=0.001)
nn_validation.fit(features_test, target_test)
nn_test_predict = nn_validation.predict(features_test)

lasso_test_score = [mean_squared_error(target_test, lasso_test_predict),
                    mean_squared_error(target_test, nn_test_predict),
                    mean_squared_error(target_test, features_test[0]),
                    mean_squared_error(target_test, features_test[1]),
                    mean_squared_error(target_test, features_test[2])]

print(lasso_test_score)

################ Weekly production ################

# Weekly prodution: ~500mwh
idx_week = target.index.floor('d').value_counts()
daily_production = target.groupby(target.index.floor('d')).sum()
daily_production_adjusted = daily_production.loc[idx_week == 48].mean()
