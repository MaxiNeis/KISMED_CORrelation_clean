# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv

import keras
import statsmodels.api as sm
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple
import math
import sys
from scipy import signal
import scipy
from tensorflow import keras
from PIL import Image
import statsmodels.api as sm

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads: List[np.ndarray], fs: float, ecg_names: List[str], model_name: str = 'model_4.h5',
                   is_binary_classifier: bool = False) -> List[Tuple[str, str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben,
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

    # ------------------------------------------------------------------------------
    # Euer Code ab hier
    # with open(model_name, 'rb') as f:
    model = keras.models.load_model(model_name)  # Lade simples Model (1 Parameter)
    # Initialisierung des QRS-Detektors

    predictions = list()
    minimum = 2714
    lowess = sm.nonparametric.lowess
    sos = signal.butter(10, 50, 'low', fs=300, output='sos')



    for idx, ecg_lead in enumerate(ecg_leads):
        print(idx)
        anzahl = len(ecg_lead)
        faktorceil = int(np.ceil(anzahl / minimum))
        if faktorceil > 1:
            biggersize = faktorceil * minimum - anzahl
            overlapping = np.ceil(biggersize / (faktorceil - 1))
            x = np.zeros((faktorceil, minimum))
            for j in range(0, faktorceil):
                start = int(j * (minimum - overlapping))
                end = int(start + minimum)
                x[j] = ecg_lead[start: end]
                print("x[j]: ", x[j])
                print("x[j].shape: ", x[j].shape)
                smoothed = sm.nonparametric.lowess(exog=np.arange(len(x[j])), endog=x[j], frac=0.2)
                print("smoothed.shape ", smoothed.shape)
                smoothed = smoothed.T
                smoothed = smoothed[1]
                x[j]=x[j]-smoothed
                x[j]  = (x[j]  - min(x[j] ))/(max(x[j] )-min(x[j] ))
            x = x.reshape((x.shape[0], x.shape[1], 1))
            predictarray = model.predict(x)
            print(predictarray)
            value = 0
            av=0
            for d in range(0, len(x)):
                pred = predictarray[d][1]
                av = av + (pred / len(x))
                if pred >= 0.8:
                    value = value + 1
            qu = value / len(x)
            #print(qu)
            if qu >= 0.34 and av>0.6:
                predictions.append((ecg_names[idx], 'A'))
            else:
                predictions.append((ecg_names[idx], 'N'))
        else:
            x = np.zeros(minimum)
            for k in range(0, anzahl):
                x[k] = ecg_lead[k]
            print("x: ", x)
            print("x.shape: ", x.shape)
            smoothed = sm.nonparametric.lowess(exog=np.arange(len(x)), endog=x, frac=0.2)
            print("smoothed.shape ", smoothed.shape)
            smoothed = smoothed.T
            smoothed = smoothed[1]
            x = x- smoothed
            x = (x - min(x)) / (max(x) - min(x))
            x = x.reshape((1, x.shape[0], 1))
            predict = model.predict(x)
            if predict[0] < 0.5:
                predictions.append((ecg_names[idx], 'A'))
            else:
                predictions.append((ecg_names[idx], 'N'))
        if ((idx + 1) % 100) == 0:
            print(str(idx + 1) + "\t Dateien wurden verarbeitet.")

    # ------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!

 
