from src.utils import args
from src.inputs import RepresentationGenerator
from csd.parse_csd_files import PreprocessingCSD
from csd.cleaning import do_cleaning
import pandas as pd
from src.models import random_forest_classifier





if __name__ == '__main__':
    # Parse CSD output files
    if args.from_scratch:
        parser = PreprocessingCSD(save_output=True)
        csd_df = parser.build_csd_output()
        csd_df = do_cleaning(csd_df, min_sol_count=100, min_habit_count=1000, save_output=True)
        input_gen = RepresentationGenerator(csd_df, save_output=True)
        inputs = input_gen.ml_set
    else:
        print(f'loading {args.input} as inputs')
        inputs = pd.read_csv(f'./checkpoints/inputs/{args.input}_dataset.csv', index_col=0)

    # make single solvent if wanted
    if args.solvent != 'all':
        inputs = inputs[inputs['Solvent'] == f'{args.solvent}']


    # Get the correct features
    if args.mode == 'one_hot':
        features = inputs.drop(['label', 'Solvent'], axis=1)
        enc = pd.get_dummies(inputs['Solvent'])
        features = features.join(enc).reset_index(drop=True)
    elif args.mode == 'drop':
        features = inputs.drop(['label', 'Solvent'], axis=1)
    else:
        features = inputs.drop(['label'], axis=1)   # Drop the labels
        sol_desc = 9  # Add solvent descriptors
                        # Merge the two



        # remove solvent and labels from feats
    labels = inputs['label']    # get labels


    # Apply model of choice
    model = random_forest_classifier(features, labels)

    # enc = pd.get_dummies(inputs['Solvent'])
    # features = features.join(enc).reset_index(drop=True)
    #
    # model = random_forest_classifier(features, labels)



    print('done')


    # Make sure all other inputs work
    ## add image inputs
    ### add models
    #### add arguemtns to reduce the model complexity e.g single solvents.
    ##### add solvent descriptors

