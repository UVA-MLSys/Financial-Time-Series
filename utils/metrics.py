import numpy as np
from sklearn.metrics import mean_squared_log_error, r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MSLE(pred, true):
    return MSE(np.log1p(pred), np.log1p(true))

def RMSLE(pred, true):
    return np.sqrt(MSLE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def calculate_metrics(true, pred):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    rmsle = RMSLE(np.where(pred<0, 0, pred) , true)
    smape = SMAPE(pred, true)

    return mae, rmse, rmsle, smape

def calculate_results(y_true: np.ndarray, y_pred:np.ndarray):
    mae, rmse, rmsle, smape  = calculate_metrics(y_true, y_pred)
    
    return {
        'rmse': rmse, 'mae': mae, 
        'rmsle': rmsle, 'smape': smape
    }