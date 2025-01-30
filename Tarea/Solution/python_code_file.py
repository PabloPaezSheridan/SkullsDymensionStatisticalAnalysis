import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

class HistoricalPeriods(Enum):
    early_predynastic = 1,
    late_predynastic = 2

# extraccion de datos segun la muestra
def sample_data(period:HistoricalPeriods) -> list[int]:
    try:
        fullpath = os.path.abspath(os.path.join('Tarea','Statement', 'datosejercicioevaluacionanchuras.xlsx'))
        df = pd.read_excel(fullpath)
    except FileNotFoundError:
        print("Error: File datosejercicioevaluacionanchuras not found.")
        return [] 

    filas_filtradas = df[df["Época histórica"] == period.value] 

    data_list = filas_filtradas["Anchura del cráneo"].tolist() 

    return data_list

# calculo de medidas estadisticas
def show_statitics_measures(data: List[int]):
    df = pd.DataFrame({'valores': data})
    media = df['valores'].mean()
    mediana = df['valores'].median()
    moda = df['valores'].mode().iat[0]
    varianza = df['valores'].var()
    desviacion_estandar = df['valores'].std()
    asimetria = df['valores'].skew()
    curtosis = df['valores'].kurt()

    print('Media:', media)
    print('Mediana:', mediana)
    print('Moda:', moda)
    print('Varianza:', varianza)
    print('Desviación estándar:', desviacion_estandar)
    print('Asimetría:', asimetria)
    print('Curtosis:', curtosis)


# diagrama de caja de bigotes para un conjunto de datos dados
def box_diagram(data : List[int], title: str):
    plt.boxplot(data)
    plt.title(title)
    plt.ylabel("Valores")
    plt.show()

# test de Kolmogorov-Smirnov para determinar si una muestra sigue una distribucion dada (normal para el ejercio)

# organizar las invocaciones de las funciones
data = sample_data(HistoricalPeriods.early_predynastic)
print(data)

show_statitics_measures(data)

box_diagram(data, HistoricalPeriods.early_predynastic.name)

# ----------------------------------------------------------------------

import numpy as np
import scipy.stats as st

def intervalo_confianza_diff_medias(muestra1, muestra2, nivel_significancia):
    n1 = len(muestra1)
    n2 = len(muestra2)
    media1 = np.mean(muestra1)
    media2 = np.mean(muestra2)
    varianza1 = np.var(muestra1, ddof=1)  # ddof=1 para usar la varianza muestral
    varianza2 = np.var(muestra2, ddof=1)

    gl = n1 + n2 - 2  # Grados de libertad

    intervalos = {}
    t_critico = st.t.ppf(1 - nivel_significancia/2, gl)  # Valor crítico de la distribución t
    error_estandar = np.sqrt(varianza1/n1 + varianza2/n2)
    margen_error = t_critico * error_estandar
    limite_inferior = (media1 - media2) - margen_error
    limite_superior = (media1 - media2) + margen_error
    intervalos[nivel_significancia] = (limite_inferior, limite_superior)

    return intervalos

# Ejemplo de uso
muestra1 = np.array([10, 12, 14, 11, 13])
muestra2 = np.array([8, 9, 11, 10, 12])
niveles_significancia = [0.05, 0.01, 0.10]

intervalos = intervalo_confianza_diff_medias(muestra1, muestra2, niveles_significancia)

for alpha, (lim_inf, lim_sup) in intervalos.items():
  print(f"Intervalo de confianza para alpha = {alpha}: ({lim_inf:.2f}, {lim_sup:.2f})")
