import numpy as np

def estimate_average_error(target, predict) -> dict:
    target = np.array(target)
    predict = np.array(predict)
    assert target.shape == predict.shape
    
    mse = 0
    mae = 0
    counts = 0
    for t,p in zip(target, predict):
        diff = t - p
                
        counts += 1
        mae += np.sum(np.abs(diff))/2.0  # this is the same as L1Loss, summed up together and divide by counts
        mse += np.sum(diff**2)/2.0       # this is the same as MSELoss, summed up together and divide by counts, as the following line
                
    if counts == 0:
        return {"mae": float(0), "mse": float(0)}
    else:
        mse /= counts
        mae /= counts
        return {"mae": float(mae), "mse": float(mse)}