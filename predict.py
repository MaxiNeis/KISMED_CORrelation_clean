# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv

import keras.models
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
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

#------------------------------------------------------------------------------
# Euer Code ab hier  
    #with open(model_name, 'rb') as f:
    model = keras.models.load_model(model_name)        # Lade simples Model (1 Parameter)
            # Initialisierung des QRS-Detektors

    predictions = list()
    minimum = 2714

    def mediansmooth(window, sliding, array):
        r = int(np.floor((len(array) - window) / sliding) + 1)
        new_array = np.zeros(r)
        for i in range(0, r):
            subarray = np.zeros(window)
            for j in range(0, window):
                subarray[j] = array[i * sliding + j]
            median = np.median(subarray)
            new_array[i] = median
        return new_array

    def scalearray(array):
        mini = min(array)
        maxi = max(array)
        diff = maxi - mini
        for i in range(0, len(array)):
            array[i] = (array[i] - mini) / diff
        return array

    for idx,ecg_lead in enumerate(ecg_leads):
        anzahl = len(ecg_lead)
        faktorceil = int(np.ceil(anzahl / minimum))
        if faktorceil > 1:
            biggersize = faktorceil * minimum - anzahl
            overlapping = np.ceil(biggersize / (faktorceil - 1))
            ecg_data = np.zeros((faktorceil,minimum))
            for j in range(0, faktorceil):
                start = int(j * (minimum - overlapping))
                end = int(start + minimum)
                ecg_data[j] = ecg_lead[start: end]
            new_array = mediansmooth(5,4,ecg_data[0])
            ecg_smoothed = np.zeros((len(ecg_data), len(new_array)))
            predictarray = np.zeros((len(ecg_data), 2))
            boolvalue = 0
            for l in range(0, len(ecg_data)):
                ecg_smoothed[l] = mediansmooth(5,4,ecg_data[l])
                ecg_smoothed[l] = scalearray(ecg_smoothed[l])
                #ecg_smoothed[l] = ecg_smoothed[l].reshape((678, 1))
                #predictarray[l] = model.predict(ecg_smoothed[l])[0]
                #if predictarray[l] < 0.5 :
                    #boolvalue = 1
            predictarray = model.predict(ecg_smoothed.reshape((ecg_smoothed.shape[0],ecg_smoothed.shape[1],1)))
            for d in range(0, len(ecg_data)):
                if predictarray[d][0] <= 0.5:
                    boolvalue = 1
            if  boolvalue == 1 :
                predictions.append((ecg_names[idx], 'A'))
            else:
                predictions.append((ecg_names[idx], 'N'))
        else:
            # ecg_data = np.zeros(minimum)
            # for k in range(0, anzahl):
            #     ecg_data[k] = ecg_lead[k]
            # ecg_smoothed = mediansmooth(5,4,ecg_data)
            # ecg_smoothed = scalearray(ecg_smoothed)
            # predict = model.predict(ecg_smoothed)[0]
            # if predict < 0.5 :
            #     predictions.append((ecg_names[idx], 'A'))
            # else:
            #     predictions.append((ecg_names[idx], 'N'))
            predictions.append((ecg_names[idx], 'A'))

        if ((idx+1) % 100)==0:
            print(str(idx+1) + "\t Dateien wurden verarbeitet.")
            
            
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
