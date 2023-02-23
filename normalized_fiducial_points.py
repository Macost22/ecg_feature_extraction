# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:25:41 2023

@author: Melissa
"""

import numpy as np
from collections import defaultdict


# funcion para normalizar el tiempo, donde se obtiene el max, min y el tiempo normalizado
def normalizar(lista):  
    xmin = min(lista) 
    xmax=max(lista)
    for i, x in enumerate(lista):
        lista[i] = (x-xmin) / (xmax-xmin)
    return xmin,xmax,lista


# funcion para normalizar la ubicación de los puntos fiduciales
def normalizar2(xmin, xmax,x):
    x = (x-xmin) / (xmax-xmin)
    return x


def normalized_segment(fiducial_points,fs,cycle:int):
    """
    Normaliza puntos fiduciales de un ciclo de señal
    Parameters
    ----------
    fiducial_points:
    fs
    cycle

    Returns
    -------

    """
    segment=[]
    for key, value in fiducial_points.items():
        segment.append(value[cycle])
        segment_t = [index/fs for index in segment]
        tmin, tmax, segment_t_normalized = normalizar(segment_t) 
    return segment_t_normalized


def normalized_fiducial(fiducial_points, fs):
    """

    Parameters
    ----------
    fiducial_points
    fs

    Returns
    -------

    """
    segments = []
    del fiducial_points['ECG']
    del fiducial_points['Tiempo']
    
    for cycle in range(len(fiducial_points['ECG_T_Offsets'])):    
        segment_t_normalized = normalized_segment(fiducial_points, fs, cycle)
        segments.append(segment_t_normalized)       
        
    segments_dic = {}
    segments_n = np.array(segments).T
    
    for key,y in zip(fiducial_points.keys(), segments_n):
        segments_dic[key] = y

    return segments_dic


# Calculo de los puntos fiduciales normalizados
def normalized_fiducial2(Paciente, fs): 
    """Esta función permite normalizar los puntos fiduciales en tiempo y amplitud
        ----------
        Paciente:dict
                  diccionario con los puntos fiduciales del paciente a analizar
        ecg: dataframe (68,240000)
        fs: frecuencia de muestreo
                
        Return
        ----------
        P1_data, P_data, P2_data, Q_data, R_data, S_data, T1_data, T_data, T2_data: list
                   puntos fiduciales normalizados
                
    """ 
    P1_data=[]
    P_data=[]
    P2_data=[]
    Q_data=[]
    R_data=[]
    S_data=[]
    T1_data=[]
    T_data=[]
    T2_data=[]
    J_data=[]
    
    for i in range(len(Paciente['ECG_T_Offsets'])):
   
        P1=Paciente['ECG_P_Onsets'][i]
        P=Paciente['ECG_P_Peaks'][i]
        P2=Paciente['ECG_P_Offsets'][i]
        Q=Paciente['ECG_Q_Peaks'][i]
        R=Paciente['ECG_R_Peaks'][i]
        S=Paciente['ECG_S_Peaks'][i]
        T1=Paciente['ECG_T_Onsets'][i]
        T=Paciente['ECG_T_Peaks'][i]
        T2=Paciente['ECG_T_Offsets'][i]
        J=Paciente['ECG_R_Offsets'][i]

        #tiempo=Paciente['Tiempo'][P1:T2]
        tmin, tmax = P1/fs, T2/fs

        ecg=Paciente['ECG'][P1:T2]
        t = np.linspace(0,1,T2-P1)


        P1n=normalizar2(tmin, tmax,P1/fs)
        Pn=normalizar2(tmin, tmax,P/fs)
        P2n=normalizar2(tmin, tmax,P2/fs)
        Qn=normalizar2(tmin, tmax,Q/fs)
        Rn=normalizar2(tmin, tmax,R/fs)
        Sn=normalizar2(tmin, tmax,S/fs)
        T1n=normalizar2(tmin, tmax,T1/fs)
        Tn=normalizar2(tmin, tmax,T/fs)
        T2n=normalizar2(tmin, tmax,T2/fs)
        Jn=normalizar2(tmin, tmax,J/fs)

        #Puntos fiduciales normalizados del paciente 
        P1_data.append(P1n)
        P_data.append(Pn)
        P2_data.append(P2n)
        Q_data.append(Qn)
        R_data.append(Rn)
        S_data.append(Sn)
        T1_data.append(T1n)
        T_data.append(Tn)
        T2_data.append(T2n)
        J_data.append(Jn)
        
        # plt.plot(t,ecg)
        # plt.scatter(P1n, Paciente['ECG'][P1],c='red')
        # plt.scatter(Pn, Paciente['ECG'][P],c='blue')
        # plt.scatter(P2n, Paciente['ECG'][P2],c='cyan')
        # plt.scatter(Qn, Paciente['ECG'][Q],c='orange')
        # plt.scatter(Rn, Paciente['ECG'][R],c='yellow')
        # plt.scatter(Sn, Paciente['ECG'][S],c='green')
        # plt.scatter(T1n, Paciente['ECG'][T1],c='gray')
        # plt.scatter(Tn, Paciente['ECG'][T],c='pink')
        # plt.scatter(T2n, Paciente['ECG'][T2],c='purple')
        
        normalized_data = {'ECG_P_Onsets':np.array(P1_data), 'ECG_P_Peaks':np.array(P_data), 'ECG_P_Offsets':np.array(P2_data),
                         'ECG_Q_Peaks':np.array(Q_data), 'ECG_R_Peaks':np.array(R_data),'ECG_S_Peaks':np.array(S_data), 
                         'ECG_T_Onsets':np.array(T1_data), 'ECG_T_Peaks':np.array(T_data), 'ECG_T_Offsets':np.array(T2_data), 
                         'ECG_R_Offsets':np.array(J_data)}
        
    return normalized_data


def flatten(l):
    return [item for sublist in l for item in sublist]


def flatten_values(fiducial_list):
    
    flatten_fiducial_list = defaultdict(list)
    
    for person in fiducial_list:
        # you can list as many input dicts as you want here
        for key, value in person.items():
            flatten_fiducial_list[key].append(value)

    for key,value in flatten_fiducial_list.items():
        flatten_fiducial_list[key] = flatten(value)
        
    return flatten_fiducial_list

if __name__ == '__main__':

    path='C:/Users/melis/Desktop/Bioseñales/ECG_veronica/ecg_70.txt'
    #path = 'C:/Users/melis/Desktop/Bioseñales/MIMIC/MIMIC_arritmia.txt'
    ecg_70 = pd.read_csv(path, sep=" ")
    # Se transponen los datos  (68,240000) = (individuo, observaciones)
    ecg_70 = ecg_70.transpose()
    # Se modifican los índices para que sean de 0 a 67
    ecg_70.index = list(range(len(ecg_70)))
    ecg1=ecg_70.iloc[0]
    fs_signal = 2000

    from fiducial_point_detection import ecg_delineation
    fiducial_points_R, fiducial_points_nk = ecg_delineation(signal=ecg1, fs=fs_signal)
