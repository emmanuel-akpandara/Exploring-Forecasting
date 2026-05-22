"""
Synthetic e-commerce order data generator.
Produces 2 years of daily order data for a logistics operation.
Run: python generate_data.py
Output: synthetic_orders.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
n = len(dates)

# Base demand: ~26,000 orders/week / 7 days = ~3,700/day
base = 3700

# Day-of-week pattern (Mon-Tue heavier processing, weekends lighter)
dow_multiplier = {0: 1.30, 1: 1.20, 2: 1.10, 3: 1.05, 4: 1.00, 5: 0.70, 6: 0.65}
dow_effect = np.array([dow_multiplier[d.dayofweek] for d in dates])

# Annual seasonality
day_of_year = np.array([d.dayofyear for d in dates])
seasonal = 1 + 0.20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

# Q4 holiday boost (Nov-Dec)
month = np.array([d.month for d in dates])
q4_boost = np.where(np.isin(month, [11, 12]), 1.35, 1.0)

# Growth trend (15% YoY growth)
trend = 1 + (np.arange(n) / n) * 0.15

# Random promo days (~10/year)
promo_days_idx = np.random.choice(n, 20, replace=False)
promo_effect = np.ones(n)
promo_effect[promo_days_idx] = np.random.uniform(1.7, 2.3, len(promo_days_idx))

# Noise
noise = np.random.normal(1.0, 0.05, n)

orders = base * dow_effect * seasonal * q4_boost * trend * promo_effect * noise
orders = np.maximum(orders.astype(int), 0)

# Allocate to two DCs (DC1 ~60%, DC2 ~40%)
dc_split = np.random.normal(0.60, 0.03, n)
dc1 = (orders * dc_split).astype(int)
dc2 = orders - dc1

df = pd.DataFrame({
    'date': dates,
    'orders_total': orders,
    'orders_dc1': dc1,
    'orders_dc2': dc2,
    'is_promo': np.isin(np.arange(n), promo_days_idx).astype(int),
    'day_of_week': [d.day_name() for d in dates],
    'month': month
})

df.to_csv('synthetic_orders.csv', index=False)
print(f"Generated {len(df)} rows")
print(f"Avg daily orders: {df['orders_total'].mean():.0f}")
print(f"Avg weekly orders: {df['orders_total'].mean() * 7:.0f}")
print(f"Promo days: {df['is_promo'].sum()}")
print(df.head())
