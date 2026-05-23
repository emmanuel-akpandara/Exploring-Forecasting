# Exploring Forecasting

This repo is me learning time series forecasting properly. Not skimming a tutorial and calling it done, but actually sitting with the problem, building things from scratch where it helps the intuition, and writing down what I found out along the way.

The setup is a synthetic e-commerce dataset. Two years of daily order volume for a logistics operation, with a weekly rhythm, a yearly cycle, a steady growth trend, a Q4 holiday bump, and the occasional promo spike. I generated it on purpose so I would know the true structure and could check whether each model actually recovered it.

## Why I did this

I wanted to understand forecasting from the ground up instead of treating it as a black box. The plan was simple. Start by looking at the data with my own eyes, then build classical baselines, then move to a more flexible model, and at every step ask whether the extra complexity was actually paying for itself.

The recurring lesson, and honestly the most useful one, is that fancier did not mean better. A well chosen classical model held its own against everything, and one of the "smart" automated tools picked a worse model than I did by hand.

## The journey

### Notebook 01, getting to know the data

Before any modelling I just looked at the series. Plotted it raw, then decomposed it into trend, seasonal, and residual at both a weekly and a yearly period. The weekly cycle turned out to be a small fast wave and the yearly cycle a big slow one with clear Q4 spikes.

I ran the Augmented Dickey Fuller test and watched the series go from non stationary to clearly stationary after first differencing. I also built two reusable evaluation utilities here, a single chronological holdout and a walk forward generator, because I knew I would need an honest way to score every model later.

The day of week profile made the abstract concrete. Monday is the busiest day, Sunday the quietest. That is a real staffing and capacity decision falling straight out of the data.

### Notebook 02, classical baselines

This is where I built things by hand. I coded Simple Exponential Smoothing from scratch first, just to feel what the smoothing parameter really does. Then I moved to Holt Winters with weekly seasonality and grid searched the additive versus multiplicative options for trend and season.

Then came the part I did not expect. I ran `auto_arima`, the tool that is supposed to find the best ARIMA model for you, and it landed at roughly 22% MAPE. My own manually specified SARIMA, with explicit seasonal differencing at lag 7, came in around 9.8%. I refit the automated model's exact orders by hand to rule out a library bug, and it reproduced the bad result. So the gap was genuinely about model selection. The automated search had simply skipped seasonal differencing.

That was the moment the whole exercise clicked for me. Understanding the data beats trusting the default.

### Notebook 03, Prophet and regressors

Last I tried Meta's Prophet, a more flexible model that handles trend and multiple seasonalities for you. The baseline came in around 8% MAPE on the same 8 week holdout, the best result of the three notebooks.

I also tested a hypothesis. If public holidays disrupt order patterns, adding a Belgian holiday calendar as a regressor should help. It did not. The model attached a meaningful effect to Armistice Day and basically nothing to the rest. A clean reminder that not every feature you can think of is a feature worth adding, and that testing the idea is better than assuming it.

## What I came away with

| Approach | Holdout MAPE |
| --- | --- |
| Holt Winters (additive trend, multiplicative season) | about 9.3% |
| Manual SARIMA (1,1,1)(1,1,1,7) | about 9.8% |
| Auto ARIMA | about 22% |
| Prophet baseline | about 8% |

A few things stuck with me.

Look at the data before you model it. Every good decision later traced back to the exploratory work.

A strong classical baseline is hard to beat, and you should always have one before you reach for something heavier.

Automated tools are a starting point, not an answer. Knowing why a model is configured the way it is matters.

Test your feature ideas instead of trusting them. The holiday regressor felt obvious and turned out not to matter here.

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

## Where this goes next

The natural continuation is machine learning methods, gradient boosted trees with lag features and maybe a sequence model, scored against these same baselines on the same holdout. The point of the exercise stays the same. Earn every bit of added complexity.
