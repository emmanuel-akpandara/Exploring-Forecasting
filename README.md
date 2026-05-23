# Exploring Forecasting

The setup is a synthetic e-commerce dataset. Two years of daily order volume for a logistics operation, with a weekly rhythm, a yearly cycle, a steady growth trend, a Q4 holiday bump, and the occasional promo spike. I generated it on purpose so I would know the true structure and could check whether each model actually recovered it.

## Why I did this

I wanted to understand forecasting from the ground up instead of treating it as a black box. The plan was simple. Start by looking at the data with my own eyes, then build classical baselines, then move to a more flexible model, and at every step ask whether the extra complexity was actually paying for itself.

The recurring lesson, and honestly the most useful one, is that fancier did not mean better. A well chosen classical model held its own against everything, and one of the "smart" automated tools picked a worse model than I did by hand.

## The journey

### Notebook 01, getting to know the data

### Notebook 02, classical baselines

### Notebook 03, Prophet and regressors

## What I came away with

| Approach | Holdout MAPE |
| --- | --- |
| Holt Winters (additive trend, multiplicative season) | about 9.3% |
| Manual SARIMA (1,1,1)(1,1,1,7) | about 9.8% |
| Auto ARIMA | about 22% |
| Prophet baseline | about 8% |


## Repo layout

```
files/        synthetic data and the generator script
lib/          forecasting error metrics (MAPE, WMAPE, MAE, RMSE, bias, SMAPE)
notebooks/    01 exploratory analysis, 02 classical baselines, 03 Prophet
figures/      saved plots
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib statsmodels pmdarima prophet
python files/generate_data.py
jupyter lab
```

Then work through the notebooks in order. Each one builds on the one before it.

