import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Carica il dataset
dataset = pd.read_csv('Mobile.csv')

pd.set_option('display.max_columns', None)

# Effettua una copia del dataset originale
copyDataset = dataset.copy()

# Elimina le righe con dati mancanti
copyDataset.dropna(inplace=True)

# Elimina i duplicati
copyDataset.drop_duplicates(inplace=True)

print(copyDataset)

# Normalizza le feature numeriche con MinMaxScaler
scaler = MinMaxScaler()
numerical_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'm_dep', 'mobile_wt']
copyDataset[numerical_features] = scaler.fit_transform(copyDataset[numerical_features])

# Elimina le feature irrilevanti
dropped_features = ['touch_screen', 'wifi', 'blue', 'dual_sim', 'four_g', 'three_g', 'talk_time', 'sc_h', 'sc_w']
copyDataset = copyDataset.drop(columns=dropped_features)

print(copyDataset.to_string(max_rows=None))

# Salva il dataset con tutte le modifiche in un nuovo file CSV
copyDataset.to_csv('dataset_modificato.csv', index=False)

# Conta le occorrenze di ogni categoria di score
score_counts = copyDataset['price_range'].value_counts()

# Crea il grafico a torta
plt.figure(figsize=(8, 8))
plt.pie(score_counts, labels=score_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribuzione delle categorie di prezzo')
plt.axis('equal')  # Per fare in modo che il grafico sia circolare
plt.show()


