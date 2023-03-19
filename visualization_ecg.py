import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import MultipleLocator


# Se define la función plot_ecg que grafica los electrocardiogramas con los colores que corresponden a cada etiqueta
def plot_ecg(tiempo,ecg,titulo):
        
        """
        Esta función permite graficar un ECG, con el color adecuado que representa su etiqueta, 
        el grid con las dimensiones del estandar médico de ECG.
        
        Parámetros
        -----------
        Tiempo: array
                Vector de tiempo utilizado para visualizar el eje x, con fs=2000
        ecg: dataframe
                Electrocardiogramas que se desean observar (individuos, observaciones)
        color: diccionario
                Diccionario donde el Key es el número de la etiqueta y Value es el color que está tendrá
        label: int
                corresponde al número de etiqueta que identifica al ECG que se está graficando
        titulo: string
                Texto con el titulo que debe llevar la gráfica
        Retur
        -----------
        plot
        """
        ax = subplot(111)
        plot(tiempo,ecg) 
        
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        
        ax.xaxis.grid(True,'minor',linewidth=0.5)
        ax.yaxis.grid(True,'minor',linewidth=0.5)
        ax.xaxis.grid(True,'major',color="darkgray",linewidth=0.8)
        ax.yaxis.grid(True,'major',color="darkgray",linewidth=0.8)
     
        ax.set_title(titulo)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude [mV]')
        


# Se crea funcion plot_original_ecg
def plot_original_ecg(ecg,t_start,t_end,fs):
        """
        Esta función permite graficar los ECG originales sin filtrar y promediar, 
        con el color adecuado que representa su etiqueta, 
        
        Parámetros
        -----------
        ecg_df: dataframe
                Electrocardiogramas que se desean observar (individuos, observaciones)
        etiquetas: dataframe (68,1)
                  Dataframe con la lista de etiquetas generadas por clasificacion.py
        lista_sujetos: list
                Lista con los índices de los sujetos a los que deseo graficar el ECG
        segundos: float
                número de segundos que se desea visualizar de los ECG
        fs: frecuencia de muestreo
        Return
        -----------
        plot
        """
        # Se escala el tamaño del plot
        # Tiempo para ecg_70 de personas sanas
        #tiempo=np.linspace(0,120,240000) 
        
        #Tiempo para casos de arritmia
        n_valores = len(ecg)
        stop = n_valores/fs
        #tiempo = np.linspace(1,n_valores,n_valores)/fs
        tiempo = np.linspace(0,stop,n_valores)

        segundos = int(t_end - t_start)
        factor = 2
        plt.rcParams['figure.figsize'] = [segundos*5*factor, 2*factor]

        # Se llama la función plot_ecg para visualizar el ECG del sujeto
        titulo= "ECG ORIGINAL "
        
        t_start = int(t_start*fs)
        t_end = int(t_end*fs)
        
        plot_ecg(tiempo[t_start:t_end],ecg[t_start:t_end],titulo)

# Se crea la función plot_ecg_fiducial_points
def plot_ecg_fiducial_points(fiducial,t_start,t_end,fs, titulo):
    """Esta función permite graficar los ECG (filtrados y promediados) con sus respectivos 
        puntos fiduaciales.
        
        Parámetros
        ----------
        fiducial:json
                 archivo generado con el codigo mainECG_generalizable que llama la 
                 función fiducial_point.r. 
        etiquetas:dataframe (68,1)
                  Dataframe con la lista de etiquetas generadas por clasificacion.py
        lista_sujetos: list
                Lista con los índices de los sujetos a los que deseo graficar el ECG
        segundos: float
                número de segundos que se desea visualizar de los ECG
        fs: frecuencia de muestreo
                
        Return
        ---------
        plot
                
    """
    #tiempo
    tiempo = fiducial['Tiempo']
    # Se escala el tamaño del plot
    segundos = int(t_end - t_start)
    factor = 2
    plt.rcParams['figure.figsize'] = [segundos*5*factor, 2*factor]
    # Diccionario que almacena el formato y color con el que se desea visualizar el punto fiducial
    dic={"ECG_P_Peaks":['o','red'],"ECG_Q_Peaks":["^","green"],"ECG_R_Peaks":['o','green'],
         "ECG_S_Peaks":['v','green'],"ECG_T_Peaks":['o','blue'], "ECG_T_Onsets":['^','blue'],
         "ECG_T_Offsets":['v','blue'],"ECG_P_Onsets":['^','red'],"ECG_P_Offsets":['v','red'],"ECG_R_Offsets":['^','cyan']}
    
    # ECG del sujeto en fiducial
    ecg = fiducial["ECG"]
    # Se llama la función plot_ecg para visualizar el ECG del sujeto    
    t_start = int(t_start*fs)
    t_end = int(t_end*fs)
    plot_ecg(tiempo[t_start:t_end],ecg[t_start:t_end],titulo)

    for key in fiducial.keys(): 
        if key != "Tiempo" and key !="ECG" and key != "ECG_R_Onsets":
            for index in fiducial[key]:
                y = index
                x = y/fs
                if t_start < y < t_end:
                    plt.scatter(x, ecg[y],marker=dic[key][0],color=dic[key][1])                       
                               

                     



