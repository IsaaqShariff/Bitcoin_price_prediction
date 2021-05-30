import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import pandas_datareader as web
import datetime as dt 

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

value_currency='INR'
crypto='BTC'

start= dt.datetime(2016,1,1)
end=dt.datetime.now()

data= web.DataReader(f'{crypto}-{value_currency}', 'yahoo', start, end)

#Preparinf data for Neural Network to analyze

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['Close'].values.reshape(-1,1))

analyze_days= 60
target_day=120 

x_train ,y_train= [], []

for x in range(analyze_days, len(scaled_data)-target_day):
    x_train.append(scaled_data[x-analyze_days:x,0])
    y_train.append(scaled_data[x+target_day,0])

x_train, y_train=np.array(x_train), np.array(y_train)    
x_train= np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))


#Creating the Neural Network for CryptoCurrency price predition
model=Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Using the test values we are testing the predicted values

test_start=dt.datetime(2020,1,1)
test_end=dt.datetime.now()


test_data=web.DataReader(f'{crypto}-{value_currency}', 'yahoo', test_start, test_end)
actual_prices=test_data['Close'].values

total_dataset=pd.concat((data['Close'], test_data['Close']), axis=0)


model_inputs= total_dataset[len(total_dataset) - len(test_data)-analyze_days:].values
model_inputs=model_inputs.reshape(-1,1)
model_inputs=scaler.fit_transform(model_inputs)

x_test =[]

for x in range(analyze_days,len(model_inputs)):
    x_test.append(model_inputs[x-analyze_days:x,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

prediction_prices=model.predict(x_test)
prediction_prices=scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color='red', label='Real_prices')
plt.plot(prediction_prices,color='blue', label='Predcicted prices')
plt.title(f'{crypto} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

#printing  the predicted values by our trainied model
real_data=[model_inputs[len(model_inputs)+1-analyze_days:len(model_inputs)+1,0]]
real_data=np.array(real_data)
real_data=np.reshape(real_data,(real_data.shape[0], real_data.shape[1],1))

prediction=model.predict(real_data)
prediction=scaler.inverse_transform(prediction)
print(prediction+1)