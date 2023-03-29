import numpy as np
import pandas as pd
from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/sunflower/Korea/실험환경/prometheus/metrics_memory_data.csv', header=0)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
train, test = train_test_split(data, test_size=0.2)
# train['timestamp'] = pd.DatetimeIndex(train['timestamp'])
# test['timestamp'] = pd.DatetimeIndex(test['timestamp'])


train = train.dropna(how='any')
test = test.dropna(how='any')

le = LabelEncoder()
train['value'] = le.fit_transform(train['value'].astype(int))

train = train.groupby(['timestamp', 'container']).mean()
train = train.reset_index()
train = train.set_index('timestamp')
test = test.set_index('timestamp')


train = train[train['value']!=np.inf]
train = train[train['value']!=-np.inf]

train['value'] = train['value'].astype(int)
train.info()
print(train)


metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.5,
    'mage_weighting': 1,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 1,
    'runtime_weighting': 0.05,
}

forecast_length = 15

model = AutoTS(
    forecast_length=forecast_length,
    subset=100,
    frequency='T',
    prediction_interval=0.95,
    ensemble=['simple', 'horizontal_generalization', 'superfast'],
    models_mode='deep',
    model_list = 'all',
    max_generations=10,
    num_validations=3,
    validation_method='seasonal 1',
    transformer_list='all',
    no_negatives=True,
    metric_weighting=metric_weighting,
    prefill_na=0,
    n_jobs='auto')


model = model.import_template(
    "model_custom_v1.csv",
    method="only",
    enforce_model_list=True,
)
test = train[train['container']=='grafana']
print(test)
model.fit(test['value'])
prediction = model.predict(forecast_length=0.5)

fig,ax = plt.subplots(figsize=(20,3))
ax.plot(test['value'])
ax.plot(prediction.forecast)