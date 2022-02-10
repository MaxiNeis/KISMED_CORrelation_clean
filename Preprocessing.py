import numpy as np
from scipy import signal
import statsmodels.api as sm
import math
import sys

########################################################################
############### START EXTERNER CODE ####################################
########################################################################
########################################################################
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

    for idx in Pvec:  # loop over all possible differences: s - t
        # do summation over p - Eq.3 in Darbon
        k = np.array(range(N))
        kplus = k + idx
        igood = np.where((kplus >= 0) & (kplus < N))  # ignore OOB data; we could also handle it
        SSD = np.zeros(len(k))
        SSD[igood] = (signal[k[igood]] - signal[kplus[igood]]) ** 2
        Sdx = np.cumsum(SSD)

        for ii in range(iStart, iEnd):  # loop over all points 's'
            distance = Sdx[ii + PatchHW] - Sdx[ii - PatchHW - 1]  # Eq 4;this is in place of point - by - point MSE
            # but note the - 1; we want to icnlude the point ii - iPatchHW

            w = math.exp(-distance / h)  # Eq 2 in Darbon
            t = ii + idx  # in the papers, this is not made explicit

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


##############################################################################################################
##################################################END EXTERNER CODE ##########################################
##############################################################################################################


sos = signal.butter(10, 50, 'low', fs=300, output='sos')

def preprocessing_training(ecg_array):


    ### Butterworth-Low-Pass-Filter
    ecg_array_butterworth_filtered = signal.sosfilt(sos, ecg_array)
    ### idee für Butterworth-Filterung aus https://www.nature.com/articles/s41597-020-0386-x.pdf
    ### Erlangung der Parameter durch exploratives Ausprobieren

    ### Baseline-Wandering-Reduzierung
    baseline = sm.nonparametric.lowess(exog=np.arange(len(ecg_array_butterworth_filtered)),
                                       endog=ecg_array_butterworth_filtered, frac=0.2)
    baseline = baseline.T
    baseline = baseline[1]
    ecg_array_baseline_wandering_reduced = ecg_array_butterworth_filtered - baseline
    ###idee für Baseline-wandering-reduction mittels lowess-fit aus https://www.nature.com/articles/s41597-020-0386-x.pdf

    ### NML-Filter
    sigma_est = np.std(ecg_array_baseline_wandering_reduced)
    ecg_array_nlm_filtered = NLM_1dDarbon(ecg_array_baseline_wandering_reduced, sigma_est * 0.7, 5, 1)
    ecg_array_sorted = np.sort(ecg_array_nlm_filtered)
    ### idee für NLM-Filter aus https://www.nature.com/articles/s41597-020-0386-x.pdf

    ### Outlier-Entfernung
    quantile_0_994 = int(np.round(len(ecg_array_nlm_filtered) * 0.994, decimals=0))
    quantile_0_006 = int(np.round(len(ecg_array_nlm_filtered) * 0.006, decimals=0))
    upper_border = ecg_array_sorted[quantile_0_994]
    lower_border = ecg_array_sorted[quantile_0_006]
    for cnt in range(len(ecg_array_nlm_filtered)):
        if ecg_array_nlm_filtered[cnt] < lower_border:
            ecg_array_nlm_filtered[cnt] = lower_border
        elif ecg_array_nlm_filtered[cnt] > upper_border:
            ecg_array_nlm_filtered[cnt] = upper_border

    ### Normalisierung
    min_value_ecg = min(ecg_array_nlm_filtered)
    max_value_ecg = max(ecg_array_nlm_filtered)
    span_ecg = max_value_ecg - min_value_ecg
    ecg_array_normalized = (ecg_array_nlm_filtered - min_value_ecg) / span_ecg

    return ecg_array_normalized



def preprocessing_prediction(ecg_array):
    ### hier haben wir einige Filter weggelassen im Vergleich zum Preprocessing unserer Trainingsdaten
    ### durch das preprocessing der Daten, die vorherzusagen waren, ist die Laufzeit gestiegen, während es keinen
    # Nennenswerten Performance-Gewinn gab

    ### Butterworth-Filter
    ecg_array_butterworth_filtered = signal.sosfilt(sos, ecg_array)

    ### Normalisierung
    min_value_ecg = min(ecg_array_butterworth_filtered)
    max_value_ecg = max(ecg_array_butterworth_filtered)
    span_ecg = max_value_ecg - min_value_ecg
    ecg_array_normalized = (ecg_array_butterworth_filtered - min_value_ecg) / span_ecg

    return ecg_array_normalized