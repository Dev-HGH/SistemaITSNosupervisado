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
        
    def encontrar_mejor_ruta(self, inicio, destino):
        heap = [(0, inicio)]
        costos = {nodo: float('inf') for nodo in self.grafo}
        costos[inicio] = 0
        ruta = {nodo: None for nodo in self.grafo}

        while heap:
            costo_actual, nodo_actual = heapq.heappop(heap)

            if nodo_actual == destino:
                break

            for vecino in self.grafo[nodo_actual]:
                costo = self.grafo[nodo_actual][vecino]
                nuevo_costo = costo_actual + costo
                # Verificamos si el nuevo costo es menor al costo actual registrado.
                if nuevo_costo < costos[vecino]:
                    costos[vecino] = nuevo_costo
                    ruta[vecino] = nodo_actual
                    heapq.heappush(heap, (nuevo_costo, vecino))

        camino = []
        nodo = destino
        while nodo:
            camino.append(nodo)
            nodo = ruta[nodo]
        camino.reverse()

        return camino, costos[destino]

# Creamos una instancia del sistema de transporte.
sistema = SistemaTransporte()

# Entrenamos el modelo de clustering con las conexiones.
sistema.entrenar_modelo(n_clusters=3)

# Predecimos el cluster de una conexión específica.
origen = 'A'
destino = 'B'
cluster = sistema.predecir_cluster(origen, destino)
print(f"La conexión de {origen} a {destino} pertenece al cluster {cluster}.")

# Encontramos la mejor ruta entre dos puntos.
punto_A = 'A'
punto_B = 'I'
ruta_optima, costo_total = sistema.encontrar_mejor_ruta(punto_A, punto_B)

print(f"Mejor ruta de {punto_A} a {punto_B}: {' → '.join(ruta_optima)}")
print(f"Costo total del viaje: {costo_total:.2f} minutos")







