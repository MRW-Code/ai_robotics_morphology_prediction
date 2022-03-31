import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from tqdm import tqdm

def random_forest_classifier(features, labels, do_kfold=True):
    print(f'RUNNING RF CLASSIFIER WITH KFOLD = {do_kfold}')
    if do_kfold:
        splits = 10
        count = 0
        kfold = StratifiedKFold(n_splits=splits, shuffle=True)

        # Placeholders for metrics
        acc = np.empty(splits)

        # stratified kfold training
        for train_index, val_index in tqdm(kfold.split(features, labels), nrows=80):
            X_train, X_test = np.array(features.iloc[train_index, :]), np.array(features.iloc[val_index, :])
            y_train, y_test = np.array(labels)[train_index], np.array(labels)[val_index]
            model = RandomForestClassifier(n_estimators=100,
                                           n_jobs=-1,
                                           verbose=0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # generate metrics
            acc[count] = accuracy_score(y_test, preds)
            count += 1
        print(f'Model has {len(labels.value_counts())} classes')
        print(f'Mean Accuracy = {np.mean(acc)}')
    return None