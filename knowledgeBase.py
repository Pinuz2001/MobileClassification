import pandas as pd
from pyswip import Prolog


# Caricamento del dataset
df = pd.read_csv('dataset_modificato1.csv')

# Apre un file Prolog per scrivere
with open('telefoni.pl', 'w') as f:
    # Per ogni riga nel DataFrame
    for index, row in df.iterrows():
        # Scrivi un fatto Prolog per il telefono
        f.write(f"telefono({index}, {row['battery_power']}, {row['clock_speed']}, {row['fc']}, {row['int_memory']},"
                f"{row['m_dep']}, {row['mobile_wt']}, {row['n_cores']}, {row['pc']}, {row['px_height']},"
                f"{row['px_width']}, {row['ram']}, {row['cluster']}).\n")

    # Aggiunge ulteriori fatti e regole
    f.write("\n% Regola: telefono_cluster(Id, Cluster) significa \" Id è l'identificatore di un telefono con cluster di appartenenza di Cluster\"\n")
    f.write("telefono_cluster(Id, Cluster) :- telefono(Id, _, _, _, _, _, _, _, _, _, _, _, Cluster).\n")

    f.write("\n% Regola: telefono_con_ram(Id, Ram) significa \"Id è l'identificatore di un telefono con Ram quantità di RAM\"\n")
    f.write("telefono_con_ram(Id, Ram) :- telefono(Id, _, _, _, _, _, _, _, _, _, _, Ram, _).\n")

    f.write("\n% Regola: telefono_con_battery_power(Id, BatteryPower) significa \"Id è l'identificatore di un telefono con BatteryPower potenza della batteria\"\n")
    f.write("telefono_con_battery_power(Id, BatteryPower) :- telefono(Id, BatteryPower, _, _, _, _, _, _, _, _, _, _, _).\n")

    f.write("\n% Regola: telefono_ad_alte_prestazioni(Id) significa \"Id è l'identificatore di un telefono con almeno 0.8 di RAM e almeno 0.7 di potenza della batteria\"\n")
    f.write("telefono_ad_alte_prestazioni(Id) :- telefono(Id, BatteryPower, _, _, _, _, _, _, _, _, _, Ram, _), Ram >= 0.8, BatteryPower >= 0.7.\n")


prolog = Prolog()
prolog.consult('telefoni.pl')  # Carica la base di conoscenza

# Esecuzione query sulla kb
results = prolog.query('telefono_cluster(Id, 2.0)')
print('I telefoni che appartengono al cluster: 2.0')
for result in results:
    print(result)

results = prolog.query('telefono_con_ram(Id, 0.6127739176910743)')
print('I telefoni che hanno come ram il valore: 0.6127739176910743')
for result in results:
    print(result)

results = prolog.query('telefono_con_battery_power(Id, 0.6305945223780896)')
print('I telefoni che hanno come battery_power il valore: 0.6305945223780896')
for result in results:
    print(result)

results = prolog.query('telefono_ad_alte_prestazioni(Id)')
print('I telefoni ad alte prestazioni:')
for result in results:
    print(result)
