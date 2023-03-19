# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:48:14 2022

@author: Melissa
"""

import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from ecg_feature_extraction.visualization_ecg import plot_ecg_fiducial_points
import matplotlib.pyplot as plt
import neurokit2 as nk


def normalization(signal):
    """
    Normalización de la señal de ECG
    Parameters
    ----------
    signal: list
        Señal de ECG

    Returns
    -------
    list
        Señal normalizada
    """
    signal_max, signal_min = np.max(signal), np.min(signal)
    signal_normalized = (signal - signal_min) / (signal_max - signal_min)
    return signal_normalized


def signal_average(signal, locs_R, fs):
    """Implementación de  ensemble average para eliminar ruido en la señal
    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_R:list
        Índices de ubicación de los picos R
    fs: float
        Frecuencia de muestreo de la señal

    Returns
    -------
    array
        Señal de ECG promediada
    """

    # longitud de cada segmento que se va a extraer (750 ms)
    segment = int(round(0.75 * fs))
    # número de picos en la señal
    N = len(locs_R)
    # número de segmentos PQRST de la señal de ECG, que sera usados para promediar
    N1 = 10
    # el promedio inicia a partir del segundo beat (el primero puede no tratarse de un PQRST completo)
    N2 = 2

    PQRST = np.zeros((segment, N1))
    ECG = np.zeros((segment, N - N1 - 2))
    ECG1 = 0

    m = 0
    # El k me permite identificar en que beat estoy ubicado para extraer el 
    # segmento de 1500 muestras
    k = 1

    while k < (N - N1 - 1):
        for i in range(N1):
            start = int(locs_R[N2 + i - 1] - 0.25 * fs)
            end = int(locs_R[N2 + i - 1] + 0.5 * fs)
            PQRST[:, i] = signal[start:end]
            ECG1 += PQRST[:, i]

        ECG1 = ECG1 / N1
        ECG[:, m] = ECG1
        ECG1 = 0
        PQRST = np.zeros((segment, N1))
        N2 += 1
        m += 1
        k += 1

    ecg_average = ECG.flatten(order='F')
    return ecg_average


def find_R(signal, height, distance, fs):
    """ Encuentra los puntos R (onda R)

    Parameters
    ----------
    signal: list
        Señal de ECG
    height: float
        Mínima altura requerida de los picos R
    distance: float
        mínima distancia horizontal requerida entre picos vecinos
    fs: float
        Frecuencia de muestreo de la señal
    Returns
    -------
    list
        Lista con los índices de ubicación de los R
    """
    # En algunos casos la amplitud de los complejo QRS < height, por lo que se debe buscar que haya un
    # mínimo de QRS encontrados en la señal (40 por minuto), esto en funcion de fs y duración de la señal.

    # Numero de picos que debe tener la señal, esperando como mínimo 40 bpm.
    n_R = len(signal) * 40 / (fs * 60)

    locs_R, _ = find_peaks(signal, height=height, distance=distance)

    # Si el número de QRS (picos R) es menor que n_R, reducir height en 0.05
    while len(locs_R) < n_R:
        print("locs_R len: {} ".format(len(locs_R)))
        height -= 0.05
        print("nuevo gr_r: {}".format(height))
        locs_R, _ = find_peaks(signal, height=height, distance=distance)
        print("nuevo locs_R len: {} \n".format(len(locs_R)))

        if height < 0.05:
            break
    return locs_R


def find_S(signal, locs_R, gr2):
    """ Encuentra los puntos S (onda s)
    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_R: list
        Lista con los índices de ubicación de los R
    gr2: float
        Distancia máxima RS
    Returns
    -------
    list
        Lista con los índices de ubicación de los S
    """
    locs_S = []
    for kk in locs_R:
        start = kk
        end = int(start + gr2)
        ocs = np.argmin(signal[start:end])
        locs_S.append(ocs + kk)
    return locs_S


def find_S2(signal, locs_S, gr10):
    """ Encuentra los puntos S2 (punto J)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_S: list
        Lista con los índices de ubicación de los S
    gr10: float
        Distancia máxima SS2 (S2 - final del complejo QRS)
    Returns
    -------
    list
        Lista con los índices de ubicación de los S2
    """
    locs_S2 = []
    for kk in range(len(locs_S) - 1):
        start = locs_S[kk]
        end = int(start + gr10)
        ocs = np.argmax(signal[start:end])
        locs_S2.append(ocs + locs_S[kk])
    return locs_S2


def find_T(signal, locs_S, gr3):
    """ Encuentra los puntos T (onda T)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_S: list
        Lista con los índices de ubicación de los S
    gr3: float
        Distancia máxima ST
    Returns
    -------
    list
        Lista con los índices de ubicación de los T
    """
    locs_T = []
    for kk in range(len(locs_S) - 1):
        start = locs_S[kk]
        end = int(start + gr3)
        ocs = np.argmax(signal[start:end])
        locs_T.append(ocs + locs_S[kk])
    return locs_T


def find_Q(signal, locs_R, gr4):
    """ Encuentra los puntos Q (onda Q)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_R: list
        Lista con los índices de ubicación de los R
    gr4: float
        Distancia máxima QR
    Returns
    -------
    list
        Lista con los índices de ubicación de los Q
    """
    locs_Q = []
    for kk in locs_R:
        start = kk
        end = int(start - gr4)
        ocs = np.argmin(signal[end:start])
        locs_Q.append(end + ocs)
    return locs_Q


def find_P(signal, locs_Q, gr5):
    """ Encuentra los puntos P (onda P)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_Q: list
        Lista con los índices de ubicación de los Q
    gr5: float
        Distancia máxima PQ
    Returns
    -------
    list
        Lista con los índices de ubicación de los P
    """
    locs_P = []
    for kk in locs_Q:
        start = kk
        end = int(start - gr5)
        ocs = np.argmax(signal[end:start])
        locs_P.append(end + ocs)
    return locs_P


def find_P1(signal, locs_P, gr6):
    """ Encuentra los puntos P1 (inicio de la onda P)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_P: list
        Lista con los índices de ubicación de los P
    gr6: float
        Distancia máxima PP1 (P1 - inicio de onda P)

    Returns
    -------
    list
        Lista con los índices de ubicación de los P1
    """
    locs_P1 = []
    for kk in locs_P:
        start = kk
        end = int(start - gr6)
        ocs = np.argmin(signal[end:start])
        locs_P1.append(end + ocs)
    return locs_P1


def find_P2(signal, locs_P, gr7):
    """ Encuentra los puntos P2 (final de la onda P)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_P: list
        Lista con los índices de ubicación de los P
    gr7: float
        Distancia máxima PP2 (P2 - final de onda P)

    Returns
    -------
    list
        Lista con los índices de ubicación de los P2
    """
    locs_P2 = []
    for kk in locs_P:
        start = kk
        end = int(start + gr7)
        ocs = np.argmin(signal[start:end])
        locs_P2.append(ocs + kk)
    return locs_P2


def find_T1(signal, locs_T, gr8):
    """ Encuentra los puntos T1 (inicio de la onda T)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_T: list
        Lista con los índices de ubicación de los T
    gr8: float
        Distancia máxima TT1 (T1 - inicio de onda T)

    Returns
    -------
    list
        Lista con los índices de ubicación de los T1
    """
    locs_T1 = []
    for kk in locs_T:
        start = kk
        end = int(start - gr8)
        ocs = np.argmin(signal[end:start])
        locs_T1.append(end + ocs)
    return locs_T1


def find_T2(signal, locs_T, gr9):
    """ Encuentra los puntos T2 (final de la onda T)

    Parameters
    ----------
    signal: list
        Señal de ECG
    locs_T: list
        Lista con los índices de ubicación de los T
    gr9: float
        Distancia máxima TT2 (T2 - final de onda T)

    Returns
    -------
    list
        Lista con los índices de ubicación de los T2
    """
    locs_T2 = []
    for kk in locs_T:
        start = kk
        end = int(start + gr9)
        ocs = np.argmin(signal[start:end])
        locs_T2.append(ocs + kk)
    return locs_T2


def find_fiducial_points_neurokit2(signal, gr_r, fs):
    """
    Encuentra los puntos fiduciales de la señal con la librería neurokit2,
    con Discrete Wavelet Method.

    Parameters
    ----------
    signal: list
        Señal de ECG
    gr_r: float
        Mínima altura requerida de los picos R
    fs: float
        Frecuencia de muestreo de la señal

    Returns
    -------
    dict
    Diccionario donde key corresponde al nombre del punto fiducial y value
    es un array con los índices donde se ubican estos puntos, tambien entrega el
    ECG y el tiempo.
    """

    n_valores = len(signal)
    stop = n_valores / fs
    tiempo = np.linspace(0, stop, n_valores)

    # Extract R-peaks locations
    _, locs_R = nk.ecg_peaks(signal, sampling_rate=fs)

    # Delineate
    if locs_R["ECG_R_Peaks"].size == 0:
        locs_R = find_R(signal, gr_r, 0.3 * fs, fs)
        locs_R = {'ECG_R_Peaks': locs_R, 'sampling_rate': fs}

    signal_ecg, waves_peak = nk.ecg_delineate(signal, locs_R, sampling_rate=fs, method="dwt")

    waves_peak['ECG_R_Peaks'] = locs_R['ECG_R_Peaks']
    waves_peak['ECG'] = signal
    waves_peak['Tiempo'] = tiempo

    return waves_peak


def find_fiducial_points(signal, fs, dic_parameters):
    """ Encuentra los puntos fiduciales de la señal mediante la búsqueda de mínimos
        y máximos locales a determinados intervalos

    Parameters
    ----------
    signal: list
        Señal de ECG
    fs
    dic_parameters

    Returns
    -------
    dict
    Diccionario donde key corresponde al nombre del punto fiducial y value es un array con
    los índices donde se ubican estos puntos, tambien entrega ECG y tiempo.
    """

    n_valores = len(signal)
    stop = n_valores / fs
    tiempo = np.linspace(0, stop, n_valores)

    _, locs_R = nk.ecg_peaks(signal, sampling_rate=fs)
    locs_R = locs_R["ECG_R_Peaks"]

    if locs_R.size == 0:
        locs_R = find_R(signal, dic_parameters['gr_r'], 0.3 * fs, fs)

    locs_S = find_S(signal, locs_R[1:], dic_parameters['gr2'])
    locs_S2 = find_S2(signal, locs_S, dic_parameters['gr10'])
    locs_T = find_T(signal, locs_S, dic_parameters['gr3'])
    locs_Q = find_Q(signal, locs_R[1:], dic_parameters['gr4'])
    locs_P = find_P(signal, locs_Q, dic_parameters['gr5'])
    locs_P1 = find_P1(signal, locs_P, dic_parameters['gr6'])
    locs_P2 = find_P2(signal, locs_P, dic_parameters['gr7'])
    locs_T1 = find_T1(signal, locs_T, dic_parameters['gr8'])
    locs_T2 = find_T2(signal, locs_T, dic_parameters['gr9'])

    fiducial_point = {'ECG_R_Peaks': locs_R[1:], 'ECG_S_Peaks': locs_S, 'ECG_R_Offsets': locs_S2,
                      'ECG_T_Peaks': locs_T, 'ECG_Q_Peaks': locs_Q, 'ECG_P_Peaks': locs_P,
                      'ECG_P_Onsets': locs_P1, 'ECG_P_Offsets': locs_P2, 'ECG_T_Onsets': locs_T1,
                      'ECG_T_Offsets': locs_T2, 'ECG': signal, 'Tiempo': tiempo}

    return fiducial_point


def ecg_delineation(signal, fs):
    """ Ejecuta funciones secuenciales para filtrar, normalizar y
        detectar puntos fiduciales (ondas características) de la señal de ECG.

    Parameters
    ----------
    signal: list
        Señal de ECG
    fs: float
        Frecuencia de muestreo de la señal de ECG

    Returns
    -------
    fiducial_R:dict
        Diccionario con los puntos fiduciales de la señal con la técnica de R
    fiducial_nk:dict
        Diccionario con los puntos fiduciales de la señal con neurokit2
    """

    # Parametros para detección de los puntos fiduciales respecto a picos R
    dic_parameters = {'gr_r': 0.8, 'gr2': 0.1 * fs, 'gr3': 0.32 * fs, 'gr4': 0.05 * fs, 'gr5': 0.2 * fs,
                      'gr6': 0.08 * fs, 'gr7': 0.065 * fs, 'gr8': 0.1 * fs, 'gr9': 0.1 * fs, 'gr10': 0.04 * fs}

    # Filtrado de la señal con neurokit2
    signal_filtered = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")

    # Normalización de la señal
    signal_normalized = normalization(signal_filtered)

    # Extracción de puntos fiduciales de la señal con algoritmo de R
    fiducial_r = find_fiducial_points(signal_normalized, fs, dic_parameters)

    # Extracción de puntos fiduciales de la señal con algoritmo de neurokit2
    try:
        fiducial_nk = find_fiducial_points_neurokit2(signal_normalized, dic_parameters['gr_r'], fs)
    except:
        fiducial_nk = fiducial_r

    return fiducial_r, fiducial_nk, signal_filtered


def vector_fiducial(fiducial, signal_zeros):
    """ Crea un vector de la longitud de la señal de ECG donde cada número
        entero se encuentra ubicacdo en el índice del punto fiducial que se
        encontró en la señal y 0 donde no hay punto fiducial.

    Parameters
    ----------
    fiducial: dict
            con los puntos fiduciales de la señal ECG
    signal_zeros: numpy array
            contiene ceros (dtype = int), del tamaño de la señal

    Returns
    -------
    array
        contiene los numéros del 1 al 9, donde cada uno se ubica
        en el indice del punto fiducial correspondiente en la señal ECG
    """

    signal_zeros[fiducial['ECG_T_Onsets']] = int(1)
    signal_zeros[fiducial['ECG_T_Offsets']] = int(2)
    signal_zeros[fiducial['ECG_T_Peaks']] = int(3)
    signal_zeros[fiducial['ECG_P_Onsets']] = int(4)
    signal_zeros[fiducial['ECG_P_Offsets']] = int(5)
    signal_zeros[fiducial['ECG_P_Peaks']] = int(6)
    signal_zeros[fiducial['ECG_Q_Peaks']] = int(7)
    signal_zeros[fiducial['ECG_S_Peaks']] = int(8)
    signal_zeros[fiducial['ECG_R_Peaks']] = int(9)
    return signal_zeros


if __name__ == '__main__':
    # Se cargan los datos de ecg
    path_arritmia = '../ecg_ii_arrhythmia.json'
    df_arritmia = pd.read_json(path_arritmia)

    fs=250
    leng=0.3*fs
    taquicardia=df_arritmia.loc[239,'ECG_II'][0]
    """
    signal=taquicardia
    n_valores = len(signal)
    stop = n_valores / fs
    tiempo = np.linspace(0, stop, n_valores)

    # Extract R-peaks locations
    _, locs_R = nk.ecg_peaks(signal, sampling_rate=fs)
    print(locs_R)

    # Delineate
    if locs_R["ECG_R_Peaks"].size == 0:
        locs_R = find_R(signal, 0.8, 0.3 * fs, fs)
        locs_R = {'ECG_R_Peaks': locs_R, 'sampling_rate': fs}

    signal_ecg, waves_peak = nk.ecg_delineate(signal, locs_R, sampling_rate=fs, method="dwt")

    waves_peak['ECG_R_Peaks'] = locs_R['ECG_R_Peaks']
    waves_peak['ECG'] = signal
    waves_peak['Tiempo'] = tiempo
    """

    fiducial_points_R, fiducial_points_nk,signal_filtered = ecg_delineation(signal=taquicardia, fs=250)

    #import matplotlib
    #matplotlib.use("Qt5Agg")
    #t_start, t_end = 0,2
    #titulo1 = 'Neurokit'
    #titulo2 = 'Algoritmo R'
    #plot_ecg_fiducial_points(fiducial_points_nk, t_start, t_end, fs, titulo1)
    #plt.show()
    #plot_ecg_fiducial_points(fiducial_points_R, t_start, t_end, fs, titulo2)
    #plt.show()
