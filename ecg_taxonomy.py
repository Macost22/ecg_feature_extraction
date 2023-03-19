import pandas as pd
import numpy as np
from scipy.signal import peak_prominences
from ecg_feature_extraction.visualization_ecg import plot_ecg_fiducial_points, plot_original_ecg
from ecg_feature_extraction.fiducial_point_detection import ecg_delineation
import neurokit2 as nk
from wfdb import processing


# Duración de la onda P
def duracion_P(paciente, fs):
    duracion_P = []

    P1 = paciente['ECG_P_Onsets']
    P2 = paciente['ECG_P_Offsets']

    for i, j in zip(P1, P2):
        if np.isnan(i) == True or np.isnan(j) == True:
            continue
        duracion_p = (j - i) / fs
        duracion_P.append(duracion_p * 1000)

    return duracion_P


# Amplitud onda P
def amplitud_P(paciente, signal, fs):
    ECG = signal
    amplitud_P = []

    P = paciente['ECG_P_Peaks']
    P2 = paciente['ECG_P_Offsets']

    for (i, j) in zip(P, P2):
        if np.isnan(i) == True or np.isnan(j) == True:
            continue
        amplitud_p = ECG[i]
        amplitud_p2 = ECG[j]
        amplitud_P.append(amplitud_p - amplitud_p2)

    return amplitud_P


def amplitud_P2(paciente, signal):
    amplitud = peak_prominences(signal, paciente['ECG_P_Peaks'], wlen=None)
    return amplitud[0]


# Duración complejo QRS
def duracion_QRS(paciente, fs):
    duracion_QRS = []
    Q = paciente['ECG_Q_Peaks']
    S = paciente['ECG_S_Peaks']

    for i, j in zip(Q, S):
        if np.isnan(i) == True or np.isnan(j) == True:
            continue
        duracion_qrs = (j - i) / fs
        duracion_QRS.append(duracion_qrs * 1000)

    return duracion_QRS


# Amplitud T
def amplitud_T(paciente, signal, fs):
    ECG = signal
    amplitud_T = []

    T = paciente['ECG_T_Peaks']
    T2 = paciente['ECG_T_Offsets']

    for i, j in zip(T, T2):
        if np.isnan(i) == True or np.isnan(j) == True:
            continue
        amplitud_t = ECG[i]
        amplitud_t2 = ECG[j]
        amplitud_T.append(amplitud_t - amplitud_t2)

    return amplitud_T


# Bloqueo AV
# Cuando la duración del segmento PR < 200 ms
def duracion_PR(paciente, fs):
    duracion_PR = []
    P1 = paciente['ECG_P_Onsets']
    R = paciente['ECG_R_Peaks']

    for i, j in zip(P1, R):
        if np.isnan(i) == True or np.isnan(j) == True:
            continue
        duracion_pr = (j - i) / fs
        duracion_PR.append(duracion_pr * 1000)

    # duracion_PR = [x for x in duracion_PR if ~np.isnan(x)]
    return duracion_PR


# Latido atrial prematuro
# Cuando amplitud de P1 y amplitud de P2 son diferentes

def amplitud_P1_P2(paciente, signal, fs):
    amplitud_P1 = []
    amplitud_P2 = []
    ECG = signal
    P1 = paciente['ECG_P_Onsets']
    P2 = paciente['ECG_P_Offsets']

    for i, j in zip(P1, P2):
        if np.isnan(i) == True or np.isnan(j) == True:
            continue
        amplitud_p1 = ECG[i]
        amplitud_p2 = ECG[j]
        amplitud_P1.append(amplitud_p1)
        amplitud_P2.append(amplitud_p2)

    return amplitud_P1, amplitud_P2


# Calculo de la frecuencia cardíaca y duracion RR
def HR_mean(paciente, fs):
    """
    Calculo de los intervalos RR, para determinar la frecuencia cardíaca promedio de cada ECG
    
    Parámetros:
    -----------
    R = paciente a analizar
    fs = int
        Frecuencia de muestreo
    Return
    -----------
    Frecuencia cardíaca media
    """
    R = paciente['ECG_R_Peaks']

    RR = []
    HR = []
    for ind in range(len(R) - 1):
        RR.append(R[ind + 1] / fs - R[ind] / fs)
        HR.append(1 / (R[ind + 1] / fs - R[ind] / fs) * 60)
    HR_mean = round(np.mean(HR))
    RR = list(map(lambda x: x * 1000, RR))
    RR = np.round(RR, 3)
    RR = [x for x in RR if ~np.isnan(x)]
    return HR_mean, RR


def taxonomy(paciente, signal, fs):
    # La onda P debe durar menos de 120 ms
    duracionP = np.mean(duracion_P(paciente, fs))
    print('Duración onda P = {} ms'.format(round(duracionP, 2)))

    # La amplitud de la onda P debe ester entre 0.15 y 0.2 mV
    amplitudP = np.mean(amplitud_P(paciente, signal, fs))
    print('Amplitud onda P = {} mV'.format(round(amplitudP, 2)))

    # La duración del complejo QRS debe estar entre 80 y 120 ms
    duracionQRS = np.mean(duracion_QRS(paciente, fs))
    print('Duración de QRS = {} ms'.format(round(duracionQRS, 2)))

    # La amplitud de la onda T debe ser positiva
    amplitudT = np.mean(amplitud_T(paciente, signal, fs))
    print('Amplitud onda T = {} mV'.format(round(amplitudT, 2)))

    # El segmento PR debe durar menos de 200 ms 
    duracionPR = np.mean(duracion_PR(paciente, fs))
    print('Duración segmento PR = {} ms'.format(round(duracionPR, 2)))

    amplitudP1, amplitudP2 = amplitud_P1_P2(paciente, signal, fs)
    amplitudP1 = np.mean(amplitudP1)
    amplitudP2 = np.mean(amplitudP2)
    print('Amplitud P1 = {} y P2 = {} mV'.format(round(amplitudP1, 2), round(amplitudP2, 2)))

    # HRmean esta normal entre 60 y 100 ms, RR dura entre 600 y 1200 ms
    HRmean, RR = HR_mean(paciente, fs)
    print('Frecuencia cardíaca = {}'.format(round(HRmean, 2)))



    # El intervalo RR debe ser regular
    diffRR = np.diff(RR)
    sano = True
    print('\nSe encontró:')

    if duracionPR > 200:
        print('- Bloqueo AV')
        sano = False

    if (amplitudP1 - amplitudP2) > 0.05:
        print('- Latido atrial prematuro')
        sano = False

    if duracionQRS > 120:
        print('- Bloqueo de rama')
        sano = False

    if HRmean < 60:
        print('- Bradicardia')

    if HRmean > 100:
        print('- Taquicardia')
        if duracionQRS < 120:
            print('   -Taquicardia supraventricular')
        sano = False

    if sano:
        print('sano')


def temporal_ecg_features(fiducial_points, signal, fs):
    # La onda P debe durar menos de 120 ms
    duracionP = np.mean(duracion_P(fiducial_points, fs))

    # La amplitud de la onda P debe ester entre 0.15 y 0.2 mV
    amplitudP = np.mean(amplitud_P(fiducial_points, signal, fs))

    # La duración del complejo QRS debe estar entre 80 y 120 ms
    duracionQRS = np.mean(duracion_QRS(fiducial_points, fs))

    # La amplitud de la onda T debe ser positiva
    amplitudT = np.mean(amplitud_T(fiducial_points, signal, fs))

    # El segmento PR debe durar menos de 200 ms
    duracionPR = np.mean(duracion_PR(fiducial_points, fs))

    amplitudP1, amplitudP2 = amplitud_P1_P2(fiducial_points, signal, fs)
    amplitudP1 = np.mean(amplitudP1)
    amplitudP2 = np.mean(amplitudP2)

    # HRmean esta normal entre 60 y 100 ms, RR dura entre 600 y 1200 ms
    HRmean, RR = HR_mean(fiducial_points, fs)

    temporal_features = {'Duracion P [ms]': round(duracionP, 2), 'Amplitud P [mV]': round(amplitudP, 2),
                         'Duracion QRS [ms]': round(duracionQRS, 2), 'Amplitud T [mV]': round(amplitudT, 2),
                         'Duracion PR [ms]': round(duracionPR, 2), 'Heart rate [bpm]': round(HRmean, 2)}

    return temporal_features


def heart_rate(x, fiducial, corru, Fs):
    peaks = np.zeros(x.shape)
    peaks[fiducial['ECG_R_Peaks']] = 1

    corrected_peak_inds = np.where(peaks == 1)[0]
    ritmo_ecg = processing.compute_hr(len(x), corrected_peak_inds, Fs)
    ritmo_ecg_nk = nk.ecg_rate(corrected_peak_inds,sampling_rate=250)

    peaks[corru == 1] = 0

    brad = np.nan * np.ones(x.shape)
    brad[ritmo_ecg < 40] = 1
    brad[corru == 1] = np.NaN

    taq = np.nan * np.ones(x.shape)
    taq[ritmo_ecg > 140] = 1
    taq[corru == 1] = np.NaN

    return peaks, brad, taq, ritmo_ecg


if __name__ == '__main__':
    path_signals = '../ecg_ii_arrhythmia.json'
    signals = pd.read_json(path_signals)

    signal = signals.loc[239, 'ECG_II']
    signal = np.array(signal, dtype=float).reshape((len(signal[0])))
    fs = 250
    fiducial_points_R, fiducial_points_nk, signal_filtered = ecg_delineation(signal, fs)
    #features = temporal_ecg_features(fiducial_points_R, signal_filtered, fs)

    #import sys
    #sys.path.append('../Humath/Functions')
    #from Humath.Functions.AnomalyDetection_ECG import anomalydetection_ecg

    #tvent = 0.8  # duración de la ventana en segundos
    #AnoDet_ecg = anomalydetection_ecg(Fs=fs, tenvt=tvent)
    #corru_ecg = AnoDet_ecg.fit_transform(signal)

    #fiducial_points_R, fiducial_points_nk, signal_filtered = ecg_delineation(signal, fs)
    #peaks_ecg, brad_ecg, taq_ecg, ritmo = heart_rate(signal, fiducial_points_nk, corru_ecg, Fs=fs)

    #taxonomy(fiducial_points_R,signal,fs)

    #print('\n con neurokit')
    #taxonomy(fiducial_points_nk,signal,fs)

    # t_start= 0
    # t_end = 5
    # import matplotlib
    # matplotlib.use("Qt5Agg")
    # plot_ecg_fiducial_points(fiducial_points_nk,t_start,t_end,fs,'neurokit')
