from src.utils import args
from csd.factory import represent_solvents, generate_csd_output, get_api_representations
from csd.image_generator import ImageGenerator
from csd.models import random_forest_classifier, kfold_fastai
import os

if __name__ == '__main__':
    os.makedirs(f'./csd/checkpoints/inputs/{args.mode}_images', exist_ok=True)
    csd_df = generate_csd_output(min_sol_counts=100, min_habit_counts=1000, save_outputs=True)

    if args.input == 'image':
        if args.from_scratch or len(os.listdir(f'./csd/checkpoints/inputs/{args.mode}_images')) == 0:
            gen = ImageGenerator(csd_df)
        else:
            print('Using images loaded from files')
        kfold_fastai(n_splits=10)
        print('done')
    else:
        api_features_df = get_api_representations(csd_df, save_outputs=True)
        # labels = api_features_df['label']
        features, labels = represent_solvents(api_features_df)
        model = random_forest_classifier(features, labels)

    print('done')


    # Make sure all other no image inputs work - currently they dont
    ## Single solvent added test for images - desc works
    ### built the robot code into the project

