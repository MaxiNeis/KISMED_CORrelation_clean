# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
from Skalpell import skalpell
from Preprocessing import preprocessing_prediction
import keras
import numpy as np
from typing import List, Tuple
from scipy import signal
from tensorflow import keras
import statsmodels.api as sm



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads: List[np.ndarray], fs: float, ecg_names: List[str], model_name: str = 'model.npy',
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
        ecg_subarrays, is_onedimensional = skalpell(ecg_lead)

        ### mehrdimensionales ECG-Array
        if not is_onedimensional:
            for cnt_subarrays in range(len(ecg_subarrays)):
                ecg_subarrays[cnt_subarrays] = preprocessing_prediction(ecg_subarrays[cnt_subarrays])
            ecg_subarrays = ecg_subarrays.reshape((ecg_subarrays.shape[0], ecg_subarrays.shape[1], 1))

            ### Vorhersage des Modells
            predictarray = model.predict(ecg_subarrays)

            #### Möglichkeit zum Fine-Tuning ####

            afib_prediction_average_threshold = 0.73 #0.6  # Gibt den Durchschnitt an, der für das gesamte Array erlangt werden muss, damit das Gesamtarray als afibrial gewertet wird
            threshold_for_counting_afibs = 0.8  # Gibt die Schwelle an, ab der eine Prediction auch als affibrial gewertet wird (pro subarray)
            afib_ratio_threshold =0.0 # 0.34  # Gibt den Anteil der Subarrays an, die als afibrial gewertet werden müssen, damit das Gesamt-Array als affibrial gewertet wird

            #####################################

            counter_afibs = 0
            average_prediction = 0


            for cnt_subarrays in range(len(ecg_subarrays)):
                prediction_subarray = predictarray[cnt_subarrays][1]
                average_prediction = average_prediction + (prediction_subarray / len(ecg_subarrays))
                if prediction_subarray >= threshold_for_counting_afibs:
                    counter_afibs = counter_afibs + 1
            ratio_afib_subarrays = counter_afibs / len(ecg_subarrays)

            ### Entscheidung
            if ratio_afib_subarrays >= afib_ratio_threshold and average_prediction >= afib_prediction_average_threshold:
                predictions.append((ecg_names[idx], 'A'))
            else:
                predictions.append((ecg_names[idx], 'N'))

        ### eindimensionales ECG-Array
        else:
            ecg_subarrays = preprocessing_prediction(ecg_subarrays)
            ecg_subarrays = ecg_subarrays.reshape((1, ecg_subarrays.shape[0], 1))
            predict = model.predict(ecg_subarrays)

            ### Prediction des Modells
            if predict[0][0] < 0.5:
                predictions.append((ecg_names[idx], 'A'))
            else:
                predictions.append((ecg_names[idx], 'N'))
        if ((idx + 1) % 100) == 0:
            print(str(idx + 1) + "\t Dateien wurden verarbeitet.")

    # ------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!



