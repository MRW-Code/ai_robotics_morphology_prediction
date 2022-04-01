from src.utils import args
from src.factory import represent_solvents, generate_csd_output
import pandas as pd
from src.models import random_forest_classifier





if __name__ == '__main__':
    if args.input == 'image':

    else:
        csd_output_df = generate_csd_output(min_sol_counts=100, min_habit_counts=1000, save_outputs=True)
        labels = csd_output_df['label']
        features = represent_solvents(csd_output_df)
        model = random_forest_classifier(features, labels)

    print('done')


    # Make sure all other inputs work
    ## add image inputs
    ### add models
    #### add arguemtns to reduce the model complexity e.g single solvents.
    ##### add solvent descriptors

