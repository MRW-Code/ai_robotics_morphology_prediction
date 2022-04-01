from src.utils import args
from src.factory import represent_solvents, generate_csd_output, get_api_representations
import pandas as pd
from src.models import random_forest_classifier





if __name__ == '__main__':
    csd_df = generate_csd_output(min_sol_counts=100, min_habit_counts=1000, save_outputs=True)

    if args.input == 'image':
        NotImplementedError()
    else:
        api_features_df = get_api_representations(csd_df, save_outputs=True)
        labels = api_features_df['label']
        features = represent_solvents(api_features_df)
        model = random_forest_classifier(features, labels)

    print('done')


    # Make sure all other inputs work
    ## add image inputs
    ### add models
    #### add arguemtns to reduce the model complexity e.g single solvents.
    ##### add solvent descriptors

