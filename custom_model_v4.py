#pip install autokeras==1.0.19 --no-deps
import pandas as pd
import tensorflow as tf

import autokeras as ak
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

dataset = pd.read_csv('/Users/sunflower/Korea/실험환경/python/data/cpu_memory_usage_v1.csv', header=0)
dataset['time'] = pd.to_datetime(dataset['time'], unit='ns')
dataset = dataset.loc[dataset.pod_name == 'drive-az2-prd-69cbb999bc-lpr67']
dataset = dataset.loc[:,['cpu_usage_nanocores', 'memory_working_set_bytes']]
dataset = dataset.dropna()


ms = MinMaxScaler()

scaled = ms.fit_transform(dataset)
dataset = pd.DataFrame(scaled, columns=['cpu_usage_nanocores','memory_working_set_bytes'])


val_split = int(len(dataset) * 0.7)
data_train = dataset[:val_split]
validation_data = dataset[val_split:]

data_x = data_train[
    [
        "cpu_usage_nanocores",
    ]
].astype("float64")

data_x_val = validation_data[
    [
        "cpu_usage_nanocores",
    ]
].astype("float64")

# Data with train data and the unseen data from subsequent time steps.
data_x_test = dataset[
    [
        "cpu_usage_nanocores",
    ]
].astype("float64")

data_y = data_train["memory_working_set_bytes"].astype("float64")

data_y_val = validation_data["memory_working_set_bytes"].astype("float64")

print(data_x.shape)  # (6549, 12)
print(data_y.shape)  # (6549,)

predict_from = 1
predict_until = 10
lookback = 3
clf = ak.TimeseriesForecaster(
    lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=1,
    objective="val_loss",
)
# Train the TimeSeriesForecaster with train data
clf.fit(
    x=data_x,
    y=data_y,
    validation_data=(data_x_val, data_y_val),
    batch_size=32,
    epochs=10,
)
# Predict with the best model(includes original training data).
predictions = clf.predict(data_x_test)
print(predictions.shape)
# Evaluate the best model with testing data.
print(clf.evaluate(data_x_val, data_y_val))
model = clf.export_model()
print(type(model)) 

try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")
print(model.summary())
# loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

# predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
# print(predicted_y)