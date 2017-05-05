import numpy as np
from sklearn import preprocessing
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor

df_train = pd.read_csv("/input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("/input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("/input/macro.csv", parse_dates=['timestamp'])

df_train.head()

y_train = df_train['price_doc'].values
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
# df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)

df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

X_all = df_values.values
print (X_all.shape)

col_mean = np.nanmean(X_all, axis=0)
inds = np.where(np.isnan(X_all))
X_all[inds] = np.take(col_mean, inds[1])

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns
print df_columns[np.where(np.isinf(col_mean))]

model = Sequential()
model.add(Dense(190, input_dim=393, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(90, init='normal', activation='relu'))
model.add(Dense(50, init='normal', activation='relu'))
model.add(Dense(50, init='normal', activation='relu'))
model.add(Dense(25, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

y_scale = preprocessing.MinMaxScaler().fit(y_train)
X_scale = preprocessing.MinMaxScaler().fit(X_all)
y_scaled = y_scale.transform(y_train)
print y_scaled
X_scaled = X_scale.transform(X_train)
reg = model.fit(X_scaled, y_scaled, nb_epoch=500)

# plt.barh(np.arange(len(df_columns))*5, reg.feature_importances_, height=3)
# plt.yticks(np.arange(len(df_columns))*5, df_columns)
# plt.show()

y_pred = y_scale.inverse_transform(np.reshape(model.predict(X_scale.transform(X_test)), (1, -1))[0])
print y_pred
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('/output/sub_nn.csv', index=False)
