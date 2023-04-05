#%%
import numpy as np
import pandas as pd
from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/sunflower/Korea/실험환경/prometheus/metrics_cpu_test_data.csv', header=0)
# data = data[data['container']=='POD']
data = data[['timestamp', 'value']]
data['value'] = data['value'].astype(float)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data = data.set_index('timestamp')
print(data)

data = data.dropna(how='all')
data = data[data['value']!=np.inf]
data = data[data['value']!=-np.inf]

data.info()
print(data)

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
    forecast_length=15,
    frequency='T',
    prediction_interval=0.95,
    ensemble=None,
    models_mode='deep',
    model_list = 'univariate',# or ['ARIMA','ETS']
    max_generations=10,
    num_validations=3,
    no_negatives=True,
    n_jobs='auto')

# future_regressor_train2d, future_regressor_forecast2d = fake_regressor(
#     data,
#     dimensions=4,
#     forecast_length=forecast_length,
#     date_col='timestamp',
#     value_col='value',
#     drop_most_recent=model.drop_most_recent,
#     aggfunc=model.aggfunc,
#     verbose=model.verbose,
# )

# TODO
# upper_limit = 0.95 
# lower_limit = np.ones((forecast_length, train.shape[1]))

# model = EventRiskForecast(
#     train,
#     forecast_length=forecast_length,
#     upper_limit=upper_limit,
#     lower_limit=lower_limit,
# )

# model = model.fit(
#     data,
#     future_regressor=future_regressor_train2d,
#     date_col='timestamp',
#     value_col='value',
# # )

# prediction = model.predict(forecast_length=30, future_regressor=future_regressor_forecast2d, verbose=0)
# forecasts_df = prediction.forecast

# model.export_template(
#     "model_custom_v2.csv",
#     models="best",
#     max_per_model_class=1,
#     include_results=True,
# )   

model = model.import_template(
    "model_custom_v2.csv",
    method="only",
    enforce_model_list=True,
)

model.fit(data['value'])

prediction = model.predict(forecast_length=15)

fig,ax = plt.subplots(figsize=(20,3))
ax.plot(data['value'])
ax.plot(prediction.forecast)

# %%
