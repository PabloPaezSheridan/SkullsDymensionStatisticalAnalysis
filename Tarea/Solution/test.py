import numpy as np
import scipy.stats as st

def intervalo_confianza_diff_medias(muestra1, muestra2, niveles_significancia):
  """
  Calcula intervalos de confianza para la diferencia de medias de dos muestras.

  Args:
    muestra1: Datos de la primera muestra.
    muestra2: Datos de la segunda muestra.
    niveles_significancia: Lista de niveles de significancia (alpha) para los cuales calcular los intervalos.

  Returns:
    Un diccionario donde las claves son los niveles de significancia y los valores son tuplas con los límites inferior y superior de los intervalos de confianza.
  """

  n1 = len(muestra1)
  n2 = len(muestra2)
  media1 = np.mean(muestra1)
  media2 = np.mean(muestra2)
  varianza1 = np.var(muestra1, ddof=1)  # ddof=1 para usar la varianza muestral
  varianza2 = np.var(muestra2, ddof=1)

  # Asumimos varianzas diferentes y muestras independientes
  gl = n1 + n2 - 2  # Grados de libertad

  intervalos = {}
  for alpha in niveles_significancia:
    t_critico = st.t.ppf(1 - alpha/2, gl)  # Valor crítico de la distribución t
    error_estandar = np.sqrt(varianza1/n1 + varianza2/n2)
    margen_error = t_critico * error_estandar
    limite_inferior = (media1 - media2) - margen_error
    limite_superior = (media1 - media2) + margen_error
    intervalos[alpha] = (limite_inferior, limite_superior)

  return intervalos

# Ejemplo de uso
muestra1 = np.array([10, 12, 14, 11, 13])
muestra2 = np.array([8, 9, 11, 10, 12])
niveles_significancia = [0.05, 0.01, 0.10]

intervalos = intervalo_confianza_diff_medias(muestra1, muestra2, niveles_significancia)

for alpha, (lim_inf, lim_sup) in intervalos.items():
  print(f"Intervalo de confianza para alpha = {alpha}: ({lim_inf:.2f}, {lim_sup:.2f})")