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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

if __name__ == '__main__':
    #### FOR CSD DATA
    csd_df = pd.read_csv('./checkpoints/inputs/mordred_descriptor_dataset.csv')
    api_features_df = get_api_representations(csd_df, save_outputs=False)
    X, labels = represent_solvents(api_features_df)

    # # #### FOR LAB DATA
    # raw_df = pd.read_csv('./lab_data/raw_data/summer_hts_data_matt.csv')
    # ml_df = LabRepresentationGenerator(raw_df, save_output=True).ml_set
    # labels = ml_df['label']
    # X = ml_df.drop('label', axis=1)

    # pca = PCA(25)
    # X = pca.fit_transform(X)

    enc_labels = pd.get_dummies(labels)
    X_lab = np.concatenate([X, pd.get_dummies(labels)], axis=1)

    # for j in range(3, 20):
    #     kmeans = KMeans(
    #         n_clusters=j, init='random',
    #         n_init=100, max_iter=500,
    #         tol=1e-04, random_state=0, verbose=0,
    #         algorithm='elkan'
    #     )
    #     # predict the labels of clusters.
    #     clusters = kmeans.fit_predict(X)
    #
    #     features = pd.DataFrame(np.concatenate([X, clusters.reshape(-1, 1)], axis=1))
    #     print(f'For model with splits = {j}')
    #     model = lab_random_forest_classifier(features, labels, splits=5)
    #
    #     print('')
    #     print('')
    #

    # scl = MinMaxScaler()
    # X = scl.fit_transform(X)

    # distortions = []
    # for i in range(1, 11):
    #     km = KMeans(
    #         n_clusters=i, init='random',
    #         n_init=10, max_iter=300,
    #         tol=1e-04, random_state=0
    #     )
    #     km.fit(X)
    #     distortions.append(km.inertia_)
    #
    # # plot
    # plt.plot(range(1, 11), distortions, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Distortion')
    # plt.show()

for j in range(1,10):

    kmeans = KMeans(
        n_clusters=j, init='random',
        n_init=100, max_iter=500,
        tol=1e-04, random_state=0, algorithm='elkan'
    )

    # predict the labels of clusters.
    clusters = kmeans.fit_predict(X_lab)

    pca = PCA(2)
    X_lab = pca.fit_transform(X_lab)

    # plotting the results:
    plt.scatter(X_lab[:, 0], X_lab[:, 1], c=clusters, s=25, cmap='viridis');
    plt.xlabel(f'{j}')
    plt.show()


