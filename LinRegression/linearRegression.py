import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AR

values = np.loadtxt('F.txt')

voice = 3

#train the autoregression model
model = AR(values[:, voice])
model_fit = model.fit()

predictions=[]
window = model_fit.k_ar #num variables
coeff = model_fit.params # Coefficients
predict_steps = 128
history = values[len(values[:, voice]) - window: , voice]

for t in range(predict_steps):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    y = coeff[0]
    for d in range(window):       
        y += coeff[d + 1] * lag[window - d - 1]
 
    predictions.append(int(y))
    history = np.append(history, y)

predictionsFull = np.zeros(shape = (len(predictions), 4))
predictionsFull[:, voice] = predictions
np.savetxt("predicted_score.txt", predictionsFull, delimiter = ' ', fmt = '%i')

