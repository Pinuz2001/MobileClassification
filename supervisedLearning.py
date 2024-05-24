import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Carica il dataset
sLdataset = pd.read_csv('dataset_modificato1.csv')

# 'cluster' contiene la feature target
X = sLdataset.drop(['cluster'], axis=1)
y = sLdataset['cluster']

# Divide il dataset in set di addestramento e test (ad esempio, 80% addestramento, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ora X_train e y_train contengono le features e le etichette del set di addestramento,
# mentre X_test e y_test contengono le features e le etichette del set di test

# Definisce i modelli
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Definisci gli iperparametri per ogni modello
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

# Esegui la ricerca degli iperparametri con la k-fold cross-validation
dt_grid_search = GridSearchCV(dt_classifier, param_grid_dt, cv=5, scoring='accuracy')
rf_grid_search = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy')

# Addestra i modelli
dt_grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)


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
dt_cv_results = cross_validate(dt_grid_search.best_estimator_, X_train, y_train, cv=5, scoring=scorers)

# Esegue la cross-validation con gli scorers personalizzati per Random Forest
rf_cv_results = cross_validate(rf_grid_search.best_estimator_, X_train, y_train, cv=5, scoring=scorers)

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


# Definisce un metodo che calcola varianza, deviazione standard e stampa il grafico circa le curve di apprendimento
def plot_learning_curves(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')

    # Calcola l'errore di addestramento e test come 1 - score
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la varianza e la deviazione standard dell'errore di addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    print(
        f"\033[95m - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    # Stampa le curve di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Training error', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Testing error', color='red')

    plt.title('Learning Curve')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


# Stampa il grafico delle curve di apprendimento per il modello Decision Tree
plot_learning_curves(dtc, X_train, y_train)

# Stampa il grafico delle curve di apprendimento per il modello Rnadom Forest
plot_learning_curves(rfc, X_train, y_train)


