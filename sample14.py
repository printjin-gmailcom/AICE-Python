from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



df_a = pd.read_json('A0007IT.json')

df_b = pd.read_csv('signal.csv')

df = pd.merge(df_a, df_b, on='RID', how='inner')



plt.figure(figsize=(10, 6))
sns.countplot(y='Address1', data=df_a, order=df_a['Address1'].value_counts().index)
plt.show()

df_cleaned = df[df['Address1'] != '-']



sns.jointplot(x='time_driving', y='speed_per_hour', data=df_a)

plt.xlabel('Time Driving (hours)')
plt.ylabel('Speed per Hour (km/h)')
plt.show()



df_temp = df[df['speed_per_hour'] <= 300]

df_temp = df_temp.drop(columns=['RID'])



df_temp.isnull().sum()

df_na = df_temp.dropna()



df_del = df_na.drop(columns=['time_departure', 'time_arrival'])



df_preset = pd.get_dummies(df_del, drop_first=True)



y = df_preset['time_driving']
X = df_preset.drop('time_driving', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)



dt = DecisionTreeRegressor(max_depth=5, min_samples_split=3, random_state=120)

rf = RandomForestRegressor(max_depth=5, min_samples_split=3, random_state=120, n_estimators=100)

dt.fit(X_train_scaled, y_train)

rf.fit(X_train_scaled, y_train)



y_pred_dt = dt.predict(X_valid_scaled)

y_pred_rf = rf.predict(X_valid_scaled)

dt_mae = mean_absolute_error(y_valid, y_pred_dt)

rf_mae = mean_absolute_error(y_valid, y_pred_rf)



model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, validation_data=(X_valid_scaled, y_valid))



epochs = range(1, len(history.history['mean_squared_error']) + 1)
train_mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_mse, label='MSE')
plt.plot(epochs, val_mse, label='Val_MSE')
plt.title('Model MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.show()