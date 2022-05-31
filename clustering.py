import pandas as pd
from src.utils import args
from csd.factory import represent_solvents, generate_csd_output, get_api_representations
from csd.image_generator import ImageGenerator
from csd.models import random_forest_classifier, kfold_fastai
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from src.utils import args
from lab_data.lab_image_generator import LabImageGenerator
from lab_data.lab_models import lab_kfold_fastai, lab_random_forest_classifier
from lab_data.lab_descriptors import LabRepresentationGenerator
import os
import re
import pandas as pd

if __name__ == '__main__':
    # #### FOR CSD DATA
    # csd_df = pd.read_csv('./checkpoints/inputs/mordred_descriptor_dataset.csv')
    # api_features_df = get_api_representations(csd_df, save_outputs=False)
    # X, labels = represent_solvents(api_features_df)

    #### FOR LAB DATA
    raw_df = pd.read_csv('./lab_data/raw_data/summer_hts_data_matt.csv')
    ml_df = LabRepresentationGenerator(raw_df, save_output=True).ml_set
    labels = ml_df['label']
    X = ml_df.drop('label', axis=1)


    enc_labels = pd.get_dummies(labels)
    X = np.concatenate([X, pd.get_dummies(labels)], axis=1)

    distortions = []
    for i in range(2, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)

    # plot
    plt.plot(range(2, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

for j in range(1,10):

    kmeans = KMeans(
        n_clusters=j, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )

    # predict the labels of clusters.
    label = kmeans.fit_predict(X)

    # plotting the results:
    plt.scatter(X[:, 0], X[:, 1], c=label, s=25, cmap='viridis');
    plt.xlabel(f'{j}')
    plt.show()


    print('done')