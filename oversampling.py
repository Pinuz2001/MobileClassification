from imblearn.over_sampling import SMOTE


def oversample_smote(dataset, target_column, random_state=42):
    # Separate features and target labels
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Create SMOTE object
    smote = SMOTE(random_state=random_state)

    # Apply SMOTE to the dataset
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

