import numpy as np


'''
Die Funktion Skalpell dient dazu, aus den gegebenen ECG-Arrays gleich lange Subarrays herzustellen 
(Auf die Länge: minimum = 2714 = InputSize des Modells).
'''


def skalpell(ecg_array):
    minimum = 2714
    is_onedimensional = True
    ecg_array_length = len(ecg_array)
    anzahl_subarrays = int(np.ceil(ecg_array_length / minimum))
    if anzahl_subarrays > 1:
        is_onedimensional = False
        overflow = anzahl_subarrays * minimum - ecg_array_length
        overlapping = np.ceil(overflow / (anzahl_subarrays - 1))
        ecg_subarrays = np.zeros((anzahl_subarrays, minimum))
        for cnt_subarray in range(anzahl_subarrays):
            start = int(cnt_subarray * (minimum - overlapping))
            end = int(start + minimum)
            ecg_subarrays[cnt_subarray] = ecg_array[start: end]
            ausgabe_array = ecg_subarrays
    else:
        ecg_onearray = np.zeros(minimum)
        for cnt_datapoint in range(ecg_array_length):
            ecg_onearray[cnt_datapoint] = ecg_array[cnt_datapoint]
        ausgabe_array = ecg_onearray
    return ausgabe_array, is_onedimensional
