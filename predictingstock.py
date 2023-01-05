import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pandas_datareader as web
import datetime as dt 

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Load Data
company = 'FB'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company, 'yahoo', start,end)

#prepare data

scaler = MinMaxScaler(feature_range=(-1,0))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaked_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#Build the model 

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of thr next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fix(x_train, y_train, epochs=25, batch_size=32)

'''Test The mModel Accuracy on Existing Data'''

#Load Test Data
test_start = dt.date(2020,1,1)
test_end = dt.datetime().now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['close'].values

total_dataset = pd.concat((data['close'],test_data['close']),axis=0)

model_inputs = total_dataset[len(total_dataset)- len(test_data)- prediction_days:].value
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Make predictions on Test Data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x, (x_.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# plot the test predictions
plt.plot(actual_prices, color="black",label=f"Actual {company} price")
plt.plot(predicted_prices, color = 'green', label=f"predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel('Time')
plt.ylabel(f'{company} share price')
plt.legend()
plt.show()