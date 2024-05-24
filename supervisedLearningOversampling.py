import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from oversampling import oversample_smote
from supervisedLearning import plot_learning_curves

# Carica il dataset
sLOdataset = pd.read_csv('dataset_modificato1.csv')

# Effettua l'oversampling del dataset
X_res, y_res = oversample_smote(sLOdataset, 'cluster')

# Definisce i modelli
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Definisce gli iperparametri per ogni modello
param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_rf = {
    'n_estimators': [10, 20, 50],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crea i classificatori
dt_classifier = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)

# Esegue la ricerca degli iperparametri con la k-fold cross-validation
dt_grid_search = GridSearchCV(dt_classifier, param_grid_dt, cv=5, scoring='accuracy', error_score='raise')
rf_grid_search = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy', error_score='raise')

# Addestra i modelli
dt_grid_search.fit(X_res, y_res)
rf_grid_search.fit(X_res, y_res)

# Crea i modelli aventi come iperparamteri quelli trovati dalla GridSearch
dtc = DecisionTreeClassifier(max_depth=dt_grid_search.best_params_['max_depth'],
                             min_samples_split=dt_grid_search.best_params_['min_samples_split'],
                             min_samples_leaf=dt_grid_search.best_params_['min_samples_leaf'])

rfc = RandomForestClassifier(n_estimators=rf_grid_search.best_params_['n_estimators'],
                             max_depth=rf_grid_search.best_params_['max_depth'],
                             min_samples_split=rf_grid_search.best_params_['min_samples_split'],
                             min_samples_leaf=rf_grid_search.best_params_['min_samples_leaf'])

# Stampa i migliori iperparametri trovati
print("Migliori iperparametri per Decision Tree:", dt_grid_search.best_params_)
print("Migliori iperparametri per Random Forest:", rf_grid_search.best_params_)


# Crea un oggetto KFold per gestire la cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definisce le metriche da utilizzare
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}

# Esegue la cross-validation con gli scorers personalizzati per Decision Tree
dt_cv_results = cross_validate(dt_grid_search.best_estimator_, X_res, y_res, cv=5, scoring=scorers)

# Esegue la cross-validation con gli scorers personalizzati per Random Forest
rf_cv_results = cross_validate(rf_grid_search.best_estimator_, X_res, y_res, cv=5, scoring=scorers)

# Stampa i risultati della cross-validation per Decision Tree
print("Decision Tree:")
print("Accuracy: %0.2f (+/- %0.2f)" % (dt_cv_results['test_accuracy'].mean(), dt_cv_results['test_accuracy'].std() * 2))
print("Precision (macro): %0.2f (+/- %0.2f)" % (dt_cv_results['test_precision_macro'].mean(), dt_cv_results['test_precision_macro'].std() * 2))
print("Recall (macro): %0.2f (+/- %0.2f)" % (dt_cv_results['test_recall_macro'].mean(), dt_cv_results['test_recall_macro'].std() * 2))
print("F1-score (macro): %0.2f (+/- %0.2f)" % (dt_cv_results['test_f1_macro'].mean(), dt_cv_results['test_f1_macro'].std() * 2))

# Stampa i risultati della cross-validation per Random Forest
print("\nRandom Forest:")
print("Accuracy: %0.2f (+/- %0.2f)" % (rf_cv_results['test_accuracy'].mean(), rf_cv_results['test_accuracy'].std() * 2))
print("Precision (macro): %0.2f (+/- %0.2f)" % (rf_cv_results['test_precision_macro'].mean(), rf_cv_results['test_precision_macro'].std() * 2))
print("Recall (macro): %0.2f (+/- %0.2f)" % (rf_cv_results['test_recall_macro'].mean(), rf_cv_results['test_recall_macro'].std() * 2))
print("F1-score (macro): %0.2f (+/- %0.2f)" % (rf_cv_results['test_f1_macro'].mean(), rf_cv_results['test_f1_macro'].std() * 2))

# Definisce le metriche e i relativi valori per Decision Tree e Random Forest
metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']
dt_scores = [dt_cv_results['test_accuracy'].mean(), dt_cv_results['test_precision_macro'].mean(),
             dt_cv_results['test_recall_macro'].mean(), dt_cv_results['test_f1_macro'].mean()]
rf_scores = [rf_cv_results['test_accuracy'].mean(), rf_cv_results['test_precision_macro'].mean(),
             rf_cv_results['test_recall_macro'].mean(), rf_cv_results['test_f1_macro'].mean()]

# Plot delle metriche per i due modelli
plt.figure(figsize=(10, 6))

plt.bar(np.arange(len(metrics))-0.1, dt_scores, width=0.1, label='Decision Tree', color='b')
plt.bar(np.arange(len(metrics))+0.1, rf_scores, width=0.1, label='Random Forest', color='g')

plt.xticks(range(len(metrics)), metrics)
plt.ylabel('Score')
plt.title('Performance Metrics')
plt.legend()
plt.show()

# Stampa il grafico delle curve di apprendimento per il modello Decision Tree
plot_learning_curves(dtc, X_res, y_res)

# Stampa il grafico delle curve di apprendimento per il modello Rnadom Forest
plot_learning_curves(rfc, X_res, y_res)

# Esegue il clustering sui dati oversampled
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_res)
clusters = kmeans.predict(X_res)

# Conta il numero di campioni in ciascun cluster
cluster_counts = pd.Series(clusters).value_counts()

# Crea e visualizza il grafico a torta per mostrare la distribuzione degli elementi in cluster
plt.figure(figsize=(8, 6))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribuzione degli elementi in cluster dopo oversampling')
plt.axis('equal')
plt.show()
