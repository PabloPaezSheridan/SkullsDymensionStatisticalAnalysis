import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import kstest
import numpy as np
import scipy.stats as st
from scipy.stats import levene

class HistoricalPeriods(Enum):
    early_predynastic = 1,
    late_predynastic = 2

# extraccion de datos segun la muestra
def sample_data(period:HistoricalPeriods) -> list[int]:
    try:
        fullpath = os.path.abspath(os.path.join('..','Statement', 'datosejercicioevaluacionanchuras.xlsx'))
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
    print('Asimetría (coeficiente de asimetría de fisher):', asimetria)
    print('Curtosis:', curtosis)

# diagrama de caja de bigotes para un conjunto de datos dados
def box_diagram(data : List[int], title: str):
    plt.boxplot(data)
    plt.title(title)
    plt.ylabel("Valores")
    plt.show()

def kolmogorov_smirnov_test(data: List[int], significante_level: float):
    data_serie = pd.Series(data)
    df = pd.DataFrame({'valores': data})

    statistic, p_value = kstest(data_serie, 'norm', args= ( df['valores'].mean(), df['valores'].std()))

    print("Estadístico KS:", statistic)
    print("Valor p:", p_value)

    if p_value > significante_level:
        print("La muestra sigue una distribución normal (no se rechaza la hipótesis nula)")
    else:
        print("La muestra no sigue una distribución normal (se rechaza la hipótesis nula)")

def intervalo_confianza_diff_medias(muestra1, muestra2, nivel_significancia):
    n1 = len(muestra1)
    n2 = len(muestra2)
    media1 = np.mean(muestra1)
    media2 = np.mean(muestra2)
    varianza1 = np.var(muestra1, ddof=1)  # ddof=1 para usar la varianza muestral
    varianza2 = np.var(muestra2, ddof=1)

    gl = n1 + n2 - 2  # Grados de libertad

    t_critico = st.t.ppf(1 - nivel_significancia/2, gl)  # Valor crítico de la distribución t
    error_estandar = np.sqrt(varianza1/n1 + varianza2/n2)
    margen_error = t_critico * error_estandar
    limite_inferior = (media1 - media2) - margen_error
    limite_superior = (media1 - media2) + margen_error
 

    return (limite_inferior, limite_superior)

def levene_test(sample1: List[int], sample2: List[int], significance_level:float):
    statistic, p_value = levene(sample1, sample2)

    print("Estadístico de Levene:", statistic)
    print("Valor p:", p_value)

    if p_value > significance_level:
        print("No se rechaza la hipótesis nula: las varianzas son iguales.")
    else:
        print("Se rechaza la hipótesis nula: las varianzas no son iguales.")
    
def t_student_test(sample1: List[int], sample2: List[int], significance_level:float):
    t_statistic, p_value = st.ttest_ind(sample1, sample2)
    
    print("Estadístico t:", t_statistic)
    print("Valor p:", p_value)

    if p_value < significance_level:
        print("Se rechaza la hipótesis nula: las medias son diferentes.")
    else:
        print("No se rechaza la hipótesis nula: no hay evidencia suficiente para decir que las medias son diferentes.")


data = sample_data(HistoricalPeriods.early_predynastic)
show_statitics_measures(data)
box_diagram(data, HistoricalPeriods.early_predynastic.name)

data = sample_data(HistoricalPeriods.late_predynastic)
show_statitics_measures(data)
box_diagram(data, HistoricalPeriods.late_predynastic.name)

data = sample_data(HistoricalPeriods.early_predynastic)
alpha = 0.05
kolmogorov_smirnov_test(data, alpha)

data = sample_data(HistoricalPeriods.late_predynastic)
alpha = 0.05
kolmogorov_smirnov_test(data, alpha)

data_early_predynastic = sample_data(HistoricalPeriods.early_predynastic)
data_late_predynastic = sample_data(HistoricalPeriods.late_predynastic)

alpha1 = 0.01
alpha2 = 0.05
alpha3 = 0.10

(lim_inf1, lim_sup1) = intervalo_confianza_diff_medias(data_early_predynastic,data_late_predynastic,alpha1)
(lim_inf2, lim_sup2) = intervalo_confianza_diff_medias(data_early_predynastic,data_late_predynastic,alpha2)
(lim_inf3, lim_sup3) = intervalo_confianza_diff_medias(data_early_predynastic,data_late_predynastic,alpha3)

print(f"Intervalo de confianza para alpha = {alpha1}: ({lim_inf1:.2f}, {lim_sup1:.2f})")
print(f"Intervalo de confianza para alpha = {alpha2}: ({lim_inf2:.2f}, {lim_sup2:.2f})")
print(f"Intervalo de confianza para alpha = {alpha3}: ({lim_inf3:.2f}, {lim_sup3:.2f})")

data_early_predynastic = sample_data(HistoricalPeriods.early_predynastic)
data_late_predynastic = sample_data(HistoricalPeriods.late_predynastic)

levene_test(data_early_predynastic,data_late_predynastic, 0.05)

data_early_predynastic = sample_data(HistoricalPeriods.early_predynastic)
data_late_predynastic = sample_data(HistoricalPeriods.late_predynastic)

t_student_test(data_early_predynastic,data_late_predynastic, 0.05)