import tensorflow as tf
import pandas as pd
from sklearn.svm._libsvm import fit

data = pd.read_csv('N:\goldh1.csv')
x = data.Education
y = data.income
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer='adam', loss='mse')
history = model.fit(x, y, epochs=5000)
