from src.utils import args
import pandas as pd
import numpy as np
from csd.parse_csd_files import PreprocessingCSD
from csd.cleaning import do_cleaning
from csd.inputs import RepresentationGenerator
from tqdm import tqdm
import os
import re

def generate_csd_output(min_sol_counts, min_habit_counts, save_outputs):
    # Parse CSD output files
    if args.from_scratch:
        parser = PreprocessingCSD(save_output=save_outputs)
        csd_df = parser.build_csd_output()
        csd_df = do_cleaning(csd_df, min_sol_count=min_sol_counts,
                             min_habit_count=min_habit_counts,
                             save_output=save_outputs)
    else:
        print('Loading CSD df from csv')
        csd_df = pd.read_csv('./csd/processed/clean_csd_output.csv', index_col=0)
    return csd_df

def get_api_representations(csd_df, save_outputs):
    if args.from_scratch:
        input_gen = RepresentationGenerator(csd_df, save_output=save_outputs, is_solvent=False)
        inputs = input_gen.ml_set
    else:
        print(f'Loading {args.input} as API molecule representations')
        inputs = pd.read_csv(f'./csd/checkpoints/inputs/{args.input}_dataset.csv', index_col=0)
    return inputs

def represent_solvents(inputs):
    # Make single solvent if desired
    if args.solvent != 'all':
        inputs = inputs[inputs['Solvent'] == f'{args.solvent}']

    # Choose solvents as descriptors, one hot or drop them altogether
    if args.mode == 'one_hot':
        labels = inputs['label']
        features = inputs.drop(['label', 'Solvent'], axis=1)
        enc = pd.get_dummies(inputs['Solvent'])
        features = features.join(enc).reset_index(drop=True)
    elif args.mode == 'drop':
        labels = inputs['label']
        features = inputs.drop(['label', 'Solvent'], axis=1)
    else:
        sol_df = pd.read_csv('./csd/raw_data/solvent_smiles.csv')
        sol_desc = RepresentationGenerator(sol_df, save_output=False, is_solvent=True).ml_set # Add solvent descriptors
        labels = inputs['label']
        features = inputs.drop(['label'], axis=1) # Drop the labels
        features = pd.merge(sol_desc, features, left_on='index', right_on='Solvent').drop(['index', 'Solvent'], axis=1)
    return features, labels

def filter_image_solvents(fnames):
    if args.solvent != 'all':
        fnames = pd.Series(fnames)
        fnames = np.array(fnames[fnames.str.contains(f'{args.solvent}')])
    return fnames

def get_aug_df():
    print('GETTING AUG DF')
    image_dir = f'./csd/checkpoints/inputs/aug_images/{args.mode}_images'
    paths = [f'{image_dir}/{x}' for x in tqdm(os.listdir(image_dir))]
    labels = [re.findall(r'.*_(.*).png', y)[0] for y in tqdm(paths)]
    model_df = pd.DataFrame({'fname': paths,
                             'label': labels})
    model_df['is_valid'] = 0
    return model_df