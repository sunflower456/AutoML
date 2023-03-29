import numpy as np
import pandas as pd
from autots import AutoTS


np.random.seed(42)
N = 100
rng = pd.date_range('2019-01-01', freq='D', periods=N)
df = pd.DataFrame(np.random.rand(N, 1), columns=['value'], index=rng)
df['value'][::7] = 10

model = AutoTS(
    forecast_length=15,
    frequency='D',
    prediction_interval=0.95,
    ensemble=None,
    models_mode='deep',
    model_list = 'univariate',# or ['ARIMA','ETS']
    max_generations=10,
    num_validations=3,
    no_negatives=True,
    n_jobs='auto')

model.fit(df['value'])
prediction = model.predict(forecast_length=30)

model.export_template(
    "model_example.csv",
    models="best",
    max_per_model_class=1,
    include_results=True,
)