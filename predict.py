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

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model_4.h5',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
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

    # ***************************************************************************
    # Copyright 2017-2019, Jianwei Zheng, Chapman University,
    # zheng120@mail.chapman.edu
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # 	http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    # Written by Jianwei Zheng.
    #(mit umbenennung idx = a)

    def NLM_1dDarbon(signal, Nvar, P, PatchHW):
        if isinstance(P, int):  # scalar has been entered; expand into patch sample index vector
            P = P - 1  # Python start index from 0
            Pvec = np.array(range(-P, P + 1))
        else:
            Pvec = P  # use the vector that has been input
        signal = np.array(signal)
        # debug = [];
        N = len(signal)

        denoisedSig = np.empty(len(signal))  # NaN * ones(size(signal));
        denoisedSig[:] = np.nan
        # to simpify, don't bother denoising edges
        iStart = PatchHW + 1
        iEnd = N - PatchHW
        denoisedSig[iStart: iEnd] = 0

        # debug.iStart = iStart;
        # debug.iEnd = iEnd;

        # initialize weight normalization
        Z = np.zeros(len(signal))
        cnt = np.zeros(len(signal))

        # convert lambda value to  'h', denominator, as in original Buades papers
        Npatch = 2 * PatchHW + 1
        h = 2 * Npatch * Nvar ** 2

        for a in Pvec:  # loop over all possible differences: s - t
            # do summation over p - Eq.3 in Darbon
            k = np.array(range(N))
            kplus = k + a
            igood = np.where((kplus >= 0) & (kplus < N))  # ignore OOB data; we could also handle it
            SSD = np.zeros(len(k))
            SSD[igood] = (signal[k[igood]] - signal[kplus[igood]]) ** 2
            Sdx = np.cumsum(SSD)

            for ii in range(iStart, iEnd):  # loop over all points 's'
                distance = Sdx[ii + PatchHW] - Sdx[ii - PatchHW - 1]  # Eq 4;this is in place of point - by - point MSE
                # but note the - 1; we want to icnlude the point ii - iPatchHW

                w = math.exp(-distance / h)  # Eq 2 in Darbon
                t = ii + a  # in the papers, this is not made explicit

                if t > 0 and t < N:
                    denoisedSig[ii] = denoisedSig[ii] + w * signal[t]
                    Z[ii] = Z[ii] + w
                    # cnt[ii] = cnt[ii] + 1
                    # print('ii',ii)
                    # print('t',t)
                    # print('w',w)
                    # print('denoisedSig[ii]', denoisedSig[ii])
                    # print('Z[ii]',Z[ii])
        # loop over shifts

        # now apply normalization
        denoisedSig = denoisedSig / (Z + sys.float_info.epsilon)
        denoisedSig[0: PatchHW + 1] = signal[0: PatchHW + 1]
        denoisedSig[- PatchHW:] = signal[- PatchHW:]
        # debug.Z = Z;

        return denoisedSig  # ,debug

    sos = signal.butter(10, 50, 'low', fs=300, output='sos')

    def filter_(ecgsignal):
        filtered = signal.sosfilt(sos, ecgsignal)
        smoothed = sm.nonparametric.lowess(exog=np.arange(len(filtered)), endog=filtered, frac=0.2)
        smoothed = smoothed.T
        smoothed = smoothed[1]
        filtered2 = np.subtract(filtered, smoothed)
        sigma_est = np.std(filtered2)
        filtered3 = NLM_1dDarbon(filtered2, sigma_est * 0.7, 5, 1)
        sort = np.sort(filtered3)
        upperBorder = int(np.round(filtered3.size * 0.994, decimals=0))
        lowerBorder = int(np.round(filtered3.size * 0.006, decimals=0))
        upperQ = sort[upperBorder]
        lowerQ = sort[lowerBorder]
        filtered4 = np.copy(filtered3)
        for i in range(len(filtered4)):
            if (filtered4[i] < lowerQ):
                filtered4[i] = lowerQ
            elif filtered4[i] > upperQ:
                filtered4[i] = upperQ
        filtered4 = (filtered4-min(filtered4))/(max(filtered4)-min(filtered4))
        return filtered4

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
                ecg_data[j] = filter_(ecg_lead[start: end])
            boolvalue = 0
            predictarray = model.predict(ecg_data.reshape((ecg_data.shape[0],ecg_data.shape[1],1)))
            for d in range(0, len(ecg_data)):
                value = 0
                if predictarray[d][0] <= 0.5:
                    value = value + 1
            qu = value/len(ecg_data)
            if qu > 0.2:
                predictions.append((ecg_names[idx], 'A'))
            else:
                predictions.append((ecg_names[idx], 'N'))
        else:
             ecg_data = np.zeros((2, minimum))
             for k in range(0, anzahl):
                 ecg_data[0][k] = ecg_lead[k]
                 ecg_data[1][k] = ecg_lead[k]
             predict = model.predict(filter_(ecg_data))[0][0]
             if predict < 0.5:
                 predictions.append((ecg_names[idx], 'A'))
             else:
                 predictions.append((ecg_names[idx], 'N'))
        if ((idx+1) % 100)==0:
            print(str(idx+1) + "\t Dateien wurden verarbeitet.")
            
            
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
