import pandas as pd
import numpy as np
from ecg_feature_extraction.fiducial_point_detection import ecg_delineation
from scipy.signal import peak_prominences

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


"""
TAXONOMY
"""


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
    print(HRmean)

    # El intervalo RR debe ser regular
    diffRR = np.diff(RR)
    sano = True

    if duracionPR > 200:
        print('Bloqueo AV \n')
        sano = False

    if (amplitudP1 - amplitudP2) > 0.05:
        print('Latido atrial prematuro \n')
        sano = False

    if duracionQRS > 120:
        print('Bloqueo de rama \n')
        sano = False

    if HRmean < 60:
        print('Bradicardia \n')

    if HRmean > 100:
        print('Taquicardia')
        if duracionQRS < 120:
            print('Taquicardia supraventricular \n')
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

    temporal_features = {'duracion P [ms]': round(duracionP, 2), 'amplitud P [mV]': round(amplitudP, 2),
                         'duracion QRS [ms]': round(duracionQRS, 2), 'amplitud T [mV]': round(amplitudT, 2),
                         'duracion PR [ms]': round(duracionPR, 2), 'Heart rate [bpm]': round(HRmean, 2)}

    return temporal_features


def load_data_arrhythmia(file_path):
    ecg = pd.read_csv(file_path, sep=" ", index_col=0)
    # Se transponen los datos (individuo, observaciones)
    ecg = ecg.transpose()
    # Se modifican los índices para que sean de 0 a N
    ecg.index = list(range(len(ecg)))
    return ecg


if __name__ == '__main__':
    # Estos tiene una fs= 250,  Wn_low = 60 y Wn_high = 0.5
    path_arritmia = 'C:/Users/melis/Desktop/Bioseñales/MIMIC/MIMIC_arritmia.txt'
    signals = load_data_arrhythmia(path_arritmia)
    # ver 1 en 0 y 5000

    signal = signals.iloc[8]
    fs = 250
    fiducial_points_R, fiducial_points_nk, signal_filtered = ecg_delineation(signal, fs)
    features = temporal_ecg_features(fiducial_points_R, signal_filtered, fs)
