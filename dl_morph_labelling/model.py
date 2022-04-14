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
from dl_morph_labelling.robot_image_augmentation import RobotImageAugmentations

def get_robot_aug_df():
    print('GETTING ROBOT AUG DF')
    image_dir = f'./dl_morph_labelling/images/aug_images'
    paths = [f'{image_dir}/{x}' for x in tqdm(os.listdir(image_dir))]
    labels = [re.findall(r'.*_(.*).png', y)[0] for y in tqdm(paths)]
    model_df = pd.DataFrame({'fname': paths,
                             'label': labels})
    model_df['is_valid'] = 0
    return model_df

def robot_train_fastai_model_classification(model_df, count):
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
    learn.fine_tune(10, cbs=[SaveModelCallback(monitor='accuracy', fname=f'./{args.no_augs}_best_cbs.pth'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=3),
                             EarlyStoppingCallback(monitor='accuracy', min_delta=0.1, patience=5)])

    os.makedirs('./dl_morph_labelling/checkpoints/figures', exist_ok=True)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.savefig(f'./dl_morph_labelling/checkpoints/figures/conf_mtrx_val_test_{count}')

    print(learn.validate())
    learn.export(f'./dl_morph_labelling/checkpoints/models/trained_model_{args.no_augs}_{count}.pkl')

def robot_kfold_fastai(robot_df, n_splits):
    print(f'Training Robot FastAI model with no_augs = {args.no_augs}')
    os.makedirs('./dl_morph_labelling/checkpoints/models/', exist_ok=True)
    paths = robot_df.fname
    labels = robot_df.label
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
            augmentor = RobotImageAugmentations()
            augmentor.do_image_augmentations(raw_model_df)
            aug_model_df = get_robot_aug_df()
            model_df = pd.concat([aug_model_df, val_df])

        trainer = robot_train_fastai_model_classification(model_df, count)
        model = load_learner(f'./dl_morph_labelling/checkpoints/models/trained_model_{args.no_augs}_{count}.pkl', cpu=False)
        best_metrics.append(model.final_record)
        count += 1

    print(best_metrics)
    print(f'mean acc = {np.mean([best_metrics[x][2] for x in range(n_splits)])}')
    return None