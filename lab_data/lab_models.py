import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import re
from fastai.vision.all import *
from src.utils import args
from lab_data.lab_image_augmentor import LabImageAugmentations
from lab_data.factory import get_lab_aug_df

def train_fastai_model_classification(model_df, count):
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col=2,
                                   item_tfms=None,
                                   batch_tfms=None,
                                   y_block=CategoryBlock(),
                                   bs=8,
                                   shuffle=True)
    metrics = [error_rate, accuracy]
    learn = cnn_learner(dls, resnet18, metrics=metrics)
    learn.fine_tune(50, cbs=[SaveModelCallback(monitor='valid_loss', fname=f'./lab_{args.no_augs}_best_cbs.pth'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=2),
                             EarlyStoppingCallback(monitor='accuracy', min_delta=0.1, patience=5)])

    os.makedirs('./lab_data/checkpoints/figures', exist_ok=True)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.savefig(f'./lab_data/checkpoints/figures/conf_mtrx_val_test_{count}')

    print(learn.validate())
    learn.export(f'./lab_data/checkpoints/models/lab_trained_model_{args.no_augs}_{count}.pkl')


def lab_kfold_fastai(n_splits):
    print(f'Training FastAI model with no_augs = {args.no_augs} and Solvent = {args.solvent}')
    os.makedirs('./lab_data/checkpoints/models/', exist_ok=True)
    paths = np.array([f'./lab_data/checkpoints/inputs/images/{file}' for file in os.listdir(f'./lab_data/checkpoints/inputs/images/')])
    labels = np.array([re.findall(r'.*_(.*).png', label)[0] for label in paths])
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    count = 0
    best_metrics = []
    for train_index, val_index in tqdm(kfold.split(paths, labels)):
        X_train, X_val = paths[train_index], paths[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        train_df = pd.DataFrame({'fname': X_train, 'label': y_train})
        train_df.loc[:, 'is_valid'] = 0
        val_df = pd.DataFrame({'fname': X_val, 'label': y_val})
        val_df.loc[:, 'is_valid'] = 1

        if args.no_augs:
            model_df = pd.concat([train_df, val_df])
        else:
            raw_model_df = pd.concat([train_df, val_df])
            augmentor = LabImageAugmentations()
            augmentor.do_image_augmentations(raw_model_df)
            aug_model_df = get_lab_aug_df()
            model_df = pd.concat([aug_model_df, val_df])

        trainer = train_fastai_model_classification(model_df, count)
        model = load_learner(f'./lab_data/checkpoints/models/lab_trained_model_{args.no_augs}_{count}.pkl', cpu=False)
        best_metrics.append(model.final_record)
        count += 1

    print(best_metrics)
    print(f'mean acc = {np.mean([best_metrics[x][2] for x in range(n_splits)])}')
    return None

def lab_random_forest_classifier(features, labels, splits):
    print(f'RUNNING RF CLASSIFIER WITH KFOLD')
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