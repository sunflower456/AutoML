import numpy as np
import pandas as pd
from autots import AutoTS

train = pd.read_csv('/Users/sunflower/Korea/실험환경/prometheus/result_cpu.csv', header=0)
test = pd.read_csv('/Users/sunflower/Korea/실험환경/prometheus/result_cpu_test.csv', header=0)
train['timestamp'] = pd.to_datetime(train['timestamp'],unit='s')
test['timestamp'] = pd.to_datetime(test['timestamp'],unit='s')
train = train.set_index('timestamp')
test = test.set_index('timestamp')

model = AutoTS(
    forecast_length=15,
    frequency='S',
    prediction_interval=0.95,
    ensemble=None,
    models_mode='deep',
    model_list = 'univariate',# or ['ARIMA','ETS']
    max_generations=10,
    num_validations=3,
    no_negatives=True,
    n_jobs='auto')

model.fit(train['value'])
prediction = model.predict(forecast_length=30)
model.export_template(
    "model.csv",
    models="best",
    max_per_model_class=1,
    include_results=True,
)   