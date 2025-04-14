# Sistema de Transporte Inteligente con Aprendizaje No Supervisado
#Parte 1

import heapq
from sklearn.cluster import KMeans
import numpy as np

class SistemaTransporte:
    def __init__(self):
        self.grafo = {
            'A': {'B': 5, 'C': 10},
            'B': {'A': 5, 'D': 7, 'E': 3},
            'C': {'A': 10, 'F': 8},
            'D': {'B': 7, 'E': 2, 'G': 6},
            'E': {'B': 3, 'D': 2, 'H': 4},
            'F': {'C': 8, 'I': 12},
            'G': {'D': 6, 'H': 5},
            'H': {'E': 4, 'G': 5, 'I': 7},
            'I': {'F': 12, 'H': 7}
        }
        self.modelo = None  # Modelo de aprendizaje no supervisado.

    def entrenar_modelo(self, n_clusters=3):
        """
        Entrena un modelo de clustering K-Means para agrupar estaciones.

        Parámetros:
        - n_clusters: Número de clusters a formar.
        """
        X = []
        estaciones = list(self.grafo.keys())
        for origen in estaciones:
            for destino, costo in self.grafo[origen].items():
                X.append([hash(origen), hash(destino), costo])  # Representación numérica de las conexiones.
        X = np.array(X)
        self.modelo = KMeans(n_clusters=n_clusters, random_state=42)
        self.modelo.fit(X)

    def predecir_cluster(self, origen, destino):
        """
        Predice el cluster al que pertenece una conexión entre dos estaciones.

        Parámetros:
        - origen: Estación de origen.
        - destino: Estación de destino.

        Retorna:
        - Cluster asignado a la conexión.
        """
        X = np.array([[hash(origen), hash(destino), self.grafo[origen][destino]]])
        return self.modelo.predict(X)[0]








