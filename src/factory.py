from src.utils import args
import pandas as pd
from csd.parse_csd_files import PreprocessingCSD
from csd.cleaning import do_cleaning
from src.inputs import RepresentationGenerator



def generate_csd_output(min_sol_counts, min_habit_counts, save_outputs):
    # Parse CSD output files
    if args.from_scratch:
        parser = PreprocessingCSD(save_output=True)
        csd_df = parser.build_csd_output()
        csd_df = do_cleaning(csd_df, min_sol_count=100, min_habit_count=1000, save_output=True)
        input_gen = RepresentationGenerator(csd_df, save_output=True, is_solvent=False)
        inputs = input_gen.ml_set
    else:
        print(f'Loading {args.input} as API molecule representations')
        inputs = pd.read_csv(f'./checkpoints/inputs/{args.input}_dataset.csv', index_col=0)
    return inputs

def represent_solvents(inputs):
    # Make single solvent if desired
    if args.solvent != 'all':
        inputs = inputs[inputs['Solvent'] == f'{args.solvent}']

    # Choose solvents as descriptors, one hot or drop them altogether
    if args.mode == 'one_hot':
        features = inputs.drop(['label', 'Solvent'], axis=1)
        enc = pd.get_dummies(inputs['Solvent'])
        features = features.join(enc).reset_index(drop=True)
    elif args.mode == 'drop':
        features = inputs.drop(['label', 'Solvent'], axis=1)
    else:
        sol_df = pd.read_csv('./csd/raw_data/solvent_smiles.csv')
        sol_desc = RepresentationGenerator(sol_df, save_output=False, is_solvent=True).ml_set # Add solvent descriptors
        features = inputs.drop(['label'], axis=1)   # Drop the labels
        features = pd.merge(sol_desc, features, left_on='index', right_on='Solvent').drop(['index', 'Solvent'], axis=1)
    return features
