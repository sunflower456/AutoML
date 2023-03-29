import numpy as np
import pandas as pd
from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/sunflower/Korea/실험환경/prometheus/metrics_memory_data.csv', header=0)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
train, test = train_test_split(data, test_size=0.2)
train['timestamp'] = pd.DatetimeIndex(train['timestamp'])
test['timestamp'] = pd.DatetimeIndex(test['timestamp'])
train = train.set_index('timestamp')
test = test.set_index('timestamp')
train = train.dropna(how='any')
test = test.dropna(how='any')


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
    prediction_interval=0.95,
    ensemble='all',
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

future_regressor_train2d, future_regressor_forecast2d = fake_regressor(
    train,
    dimensions=4,
    forecast_length=forecast_length,
    date_col='timestamp',
    value_col='value',
    drop_most_recent=model.drop_most_recent,
    aggfunc=model.aggfunc,
    verbose=model.verbose,
)

# TODO
# upper_limit = 0.95 
# lower_limit = np.ones((forecast_length, train.shape[1]))

# model = EventRiskForecast(
#     train,
#     forecast_length=forecast_length,
#     upper_limit=upper_limit,
#     lower_limit=lower_limit,
# )

model = model.fit(
    train,
    future_regressor=future_regressor_train2d,
    date_col='timestamp',
    value_col='value',
)

prediction = model.predict(forecast_length=30, future_regressor=future_regressor_forecast2d, verbose=0)
forecasts_df = prediction.forecast

model.export_template(
    "model_custom_v1.csv",
    models="best",
    max_per_model_class=1,
    include_results=True,
)   