from pgmpy.models import BayesianModel
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import BayesianEstimator


# Funzione che crea la struttura della rete bayesiana
def createBayesianNetwork(dataset):
    edges = []

    # Crea gli archi della rete bayesiana (correlazioni tra variabili)
    for column in dataset:
        if column != 'cluster':
            edges.append(('cluster', column))
    edges.append(('clock_speed', 'battery_power'))
    edges.append(('fc', 'pc'))
    edges.append(('m_dep', 'battery_power'))
    edges.append(('m_dep', 'mobile_wt'))
    edges.append(('pc', 'ram'))
    edges.append(('px_width', 'm_dep'))
    edges.append(('px_height', 'm_dep'))
    edges.append(('pc', 'battery_power'))

    # Crea il modello della rete bayesiana utilizzando la lista di archi 'edges'
    model = BayesianModel(edges)

    return model


# Carica il dataset che serve per l'addestramento della rete bayesiana
bNdataset = pd.read_csv('dataset_modificato1.csv')

# Seleziona solo le prime 500 righe
bNdataset = bNdataset.iloc[:500]

# Crea il modello della rete bayesiana
bayesian_network_model = createBayesianNetwork(bNdataset)

# Stampa della struttura della rete bayesiana
print("Struttura della rete bayesiana:")
print(bayesian_network_model.edges())

# Ottiene gli archi dalla rete bayesiana
edges = bayesian_network_model.edges()

# Crea il grafo della rete bayesiana utilizzando NetworkX
graph = nx.DiGraph()
graph.add_edges_from(edges)

# Disegna il grafico della rete bayesiana
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(graph)  # Imposta il layout del grafo
nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold")
plt.title("Grafico della rete bayesiana")
plt.show()

# Addestra la rete bayesiana utilizzando i dati nel dataset e l'estimatore BayesianEstimator
bayesian_network_model.fit(bNdataset, estimator=BayesianEstimator)

# Ora le CPD per tutte le variabili della rete bayesiana sono state apprese dai dati del dataset
# Si può accedere alle CPD di ciascuna variabile utilizzando il metodo 'get_cpds'
for node in bayesian_network_model.nodes():
    print(f"CPD of {node}:")
    print(bayesian_network_model.get_cpds(node))

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Calcola la matrice di correlazione
correlation_matrix = bNdataset.corr()

# Stampa la matrice di correlazione
print(correlation_matrix)

# Seleziona una riga casuale dal DataFrame escludendo la colonna 'cluster'
random_example = bNdataset.drop(columns='cluster').sample(1)

print(random_example)

# Usa il modello per prevedere il valore del cluster
predicted_cluster = bayesian_network_model.predict(random_example)

print(predicted_cluster)
