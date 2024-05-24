import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carica il dataset e rimuove la feature del 'price_range'
uLdataset = pd.read_csv('dataset_modificato.csv')
uLdataset = uLdataset.drop('price_range', axis=1)  # Escludi la feature del price range

# Regola del gomito
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(uLdataset)
    inertias.append(kmeans.inertia_)

# Traccia l'andamento dell'inerzia
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Numero di cluster')
plt.ylabel('Inerzia')
plt.title('Regola del Gomito')
plt.show()

optimal_k = 3  # Sostituisce con il numero ottimale di cluster scelto
kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(uLdataset)
clusters = kmeans.predict(uLdataset)

uLdataset['cluster'] = clusters

cluster_counts = {}
for label in clusters:
    cluster_counts[label] = cluster_counts.get(label, 0) + 1

# Estrae i valori dal dizionario
cluster_counts_list = list(cluster_counts.values())

# Crea un grafico a torta per visualizzare la distribuzione dei telefoni in cluster
plt.figure(figsize=(8, 6))
plt.pie(cluster_counts_list, labels=[f' Cluster {label}' for label in cluster_counts.keys()], autopct='%1.1f%%', startangle=140)
plt.title('Distribuzione dei telefoni nei cluster')
plt.axis('equal')
plt.show()

# Salva il dataset con tutte le modifiche in un nuovo file CSV
uLdataset.to_csv('dataset_modificato1.csv', index=False)











