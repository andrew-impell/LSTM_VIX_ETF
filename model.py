import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def find_all(string, substring):
    """
    Function: Returning all the index of substring in a string
    Arguments: String and the search string
    Return:Returning a list
    """
    length = len(substring)
    c = 0
    indexes = []
    while c < len(string):
        if string[c:c+length] == substring:
            indexes.append(c)
        c = c+1
    return indexes


cwd = os.getcwd()

exclude_list = ['VXX', 'VIXY', 'VIIX', 'VIXM', 'VXZ', 'XVZ', 'TVIX']


def get_etfglob(exclude):
    all_glob = set(glob(cwd + '/econ/ETFs/*.txt'))

    for etf in exclude:
        all_glob -= set(glob(cwd + f'/econ/ETFs/{etf}*'))

    return list(all_glob)


ETF_glob = glob(cwd + '/econ/ETFs/*.txt')

etf_list = []

for ETF in ETF_glob:
    first = ETF.rfind('/')
    last = find_all(ETF, '.')[-2]
    etf_name = ETF[first+1:last].upper()
    etf = pd.read_csv(ETF, header=0, delimiter=',')
    etf_clean = pd.DataFrame()
    etf_clean[f'{etf_name}'] = etf['Close']
    etf_clean['Date'] = pd.to_datetime(etf['Date'])
    etf_clean.set_index('Date', inplace=True, drop=True)
    etf_list.append(etf_clean)

ETF_df = pd.concat(etf_list, axis=1)
ETF_df = ETF_df.loc[:, ~ETF_df.columns.duplicated()]

ETF_df.drop(exclude_list, axis=1)

vix_data = pd.read_csv(cwd + '/data/vix-daily.csv', header=0)

vix_data['Date'] = pd.to_datetime(vix_data['Date'])
vix_data.set_index('Date', inplace=True, drop=True)

vix_data.close = vix_data['VIX Close']

first_date = vix_data.index.values[0]
last_date = vix_data.index.values[-1]

print(first_date, last_date)

ETF_clipped = ETF_df.loc[first_date:last_date, :]

print(f'Before clipping: {len(ETF_clipped.columns)}')

ETF_final = ETF_clipped.dropna(axis='columns', how='any', thresh=3100)

print(f'After clipping: {len(ETF_final.columns)}')

'''
mean_dict = {}

for column in ETF_final.columns[1:]:
    mean_dict[column] = ETF_final[column].mean()

sorted_mean = \
    {k: v for k, v in sorted(mean_dict.items(), key=lambda item: item[1])}
print(sorted_mean)

'''
ETF_final.fillna(0, inplace=True)

print(len(ETF_final.columns))

ETF_final['VIX'] = vix_data.close

n_steps = 30

n_features = len(ETF_final.columns) - 1
'''
# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
scaled = scaler.fit_transform(ETF_final.values)
'''

scaled = ETF_final.values
reframed = series_to_supervised(scaled, 1, 1)

values = reframed.values

train_per = 0.8
train, test = np.split(values, [int(train_per * len(values))])

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72,
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
# inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)[:, [0]]
# inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
# calculate RMSE
plt.plot(test_X[:, -1], yhat[:, -1])
plt.show()
rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)

'''
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

# plt.plot(X_test, clf.predict(X_test))

# ETF_final.loc[:, ['SPY', 'VIX']].plot()
# plt.show()
