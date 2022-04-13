from src.utils import args
from src.factory import represent_solvents, generate_csd_output, get_api_representations
from src.image_generator import ImageGenerator
from src.models import random_forest_classifier, kfold_fastai
import os
import re
import pandas as pd



if __name__ == '__main__':
    raw_df = pd.read_csv('./lab_data/raw_data/summer_hts_data_matt.csv')

    if args.input == 'image':
        raise NotImplementedError()
    else:


    #     if args.from_scratch or len(os.listdir(f'./checkpoints/inputs/{args.mode}_images')) == 0:
    #         gen = ImageGenerator(csd_df)
    #     else:
    #         print('Using images loaded from files')
    #     kfold_fastai(n_splits=10)
    #     print('done')
    # else:
    #     api_features_df = get_api_representations(csd_df, save_outputs=True)
    #     # labels = api_features_df['label']
    #     features, labels = represent_solvents(api_features_df)
    #     model = random_forest_classifier(features, labels)
    #
    # print('done')

    print('done')





    # if args.input == 'image':
    #     if args.from_scratch or len(os.listdir(f'./checkpoints/inputs/{args.mode}_images')) == 0:
    #         gen = ImageGenerator(csd_df)
    #     else:
    #         print('Using images loaded from files')
    #     kfold_fastai(n_splits=10)
    #     print('done')
    # else:
    #     api_features_df = get_api_representations(csd_df, save_outputs=True)
    #     # labels = api_features_df['label']
    #     features, labels = represent_solvents(api_features_df)
    #     model = random_forest_classifier(features, labels)
    #
    # print('done')


    # Make sure all other no image inputs work - currently they dont
    ## Single solvent added test for images - desc works
    ### built the robot code into the project
