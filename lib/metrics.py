import numpy as np

def mape(actual,forecast):
    """
    Mean Absolute Percentage Error
    It measures the average magnitude of error produced by a model
    or how far off a model's prediction is on average 

    """
    return np.mean(np.abs((actual-forecast)/actual))*100
    

def wmape(actual,forecast):
    """
    Weighted Mean Absolute Percentage Error
    it is used when the usecase requires to place certainty on
    certain sales. It gives weight on the prioritized item that
    biases prediction error towards it
    """
    return np.sum(np.abs(actual-forecast)/np.sum(actual))*100
    

def mae(actual,forecast):
    """
    Mean Absolute Error
    Absolute value difference between a model's prediction and ground truth
    averaged across the dataset
    """
    return np.mean(np.abs(actual - forecast))
    

def rmse(actual,forecast):
    """
    Root Mean Squared Error
    RMSE measures the average error magnitude, but it penalizes
    larger errors more heavily than MAE by squaring them first
    """
    return np.sqrt(np.mean((actual - forecast) ** 2))

def bias(actual,forecast):
    """
    Bias measures the direction of the forecast.
    A positive value means you are under-forecasting;
    a negative value means you are over-forecasting.
    """
    return np.mean(forecast-actual)

def bias_pct(actual,forecast):
    """
    This scales the absolute bias against the total volume of
    actual demand, showing your structural over- or under-forecasting
    tendency as a percentage.
    """
    return (np.sum(forecast-actual) / np.sum(actual)) * 100

def smape(actual,forecast):
    """
    SMAPE is a percentage-based error metric that bounds the error between 0% and 200% by dividing 
    the absolute error by the average of the actual and forecast values.
    """
    # Adding a tiny epsilon prevents a division-by-zero error if both actual and forecast are 0
    eps = 1e-9
    return np.mean(200 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + eps))



        


