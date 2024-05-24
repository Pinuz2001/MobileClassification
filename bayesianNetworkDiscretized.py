from pgmpy.models import BayesianModel
import pandas as pd
from pgmpy.estimators import BayesianEstimator
import numpy as np


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


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Carica il dataset che serve per l'addestramento della rete bayesiana
bNdataset = pd.read_csv('dataset_modificato1.csv')

# Seleziona solo le prime 500 righe
bNdataset = bNdataset.iloc[:500]

# Discretizza i valori nel tuo dataset
bNdataset_discretized = bNdataset.copy()
for column in bNdataset_discretized.columns:
    if column != 'cluster':
        bNdataset_discretized[column] = pd.cut(bNdataset_discretized[column], bins=10, labels=False)

print(bNdataset_discretized)

# Crea il modello della rete bayesiana
bayesian_network_model_discretized = createBayesianNetwork(bNdataset_discretized)

# Addestra la rete bayesiana utilizzando i dati discretizzati nel dataset e l'estimatore BayesianEstimator
bayesian_network_model_discretized.fit(bNdataset_discretized, estimator=BayesianEstimator)

# Stampa le nuove CPD per tutte le variabili
for node in bayesian_network_model_discretized.nodes():
    print(f"CPD of {node}:")
    print(bayesian_network_model_discretized.get_cpds(node))

# Ottiene i nomi delle colonne dal tuo DataFrame, escludendo 'cluster'
columns = bNdataset_discretized.drop(columns='cluster').columns

# Genera valori casuali per ciascuna caratteristica
random_values = {column: np.random.randint(0, 10) for column in columns}

# Crea un nuovo DataFrame con i valori casuali
new_example = pd.DataFrame([random_values])

print(new_example)

# Usa il modello per prevedere il valore del cluster
predicted_cluster = bayesian_network_model_discretized.predict(new_example)

print(predicted_cluster)


