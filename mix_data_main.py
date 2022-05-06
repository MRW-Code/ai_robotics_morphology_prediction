from src.utils import args
from csd.factory import represent_solvents, generate_csd_output, get_api_representations
from csd.image_generator import ImageGenerator
from csd.models import random_forest_classifier, kfold_fastai
import os
from src.utils import args
from lab_data.lab_image_generator import LabImageGenerator
from lab_data.lab_models import lab_kfold_fastai, lab_random_forest_classifier
from lab_data.lab_descriptors import LabRepresentationGenerator
import os
import re
import pandas as pd
from mix_data.mix_data_models import mix_fastai


if __name__ == '__main__':
    os.makedirs(f'./csd/checkpoints/inputs/{args.mode}_images', exist_ok=True)
    os.makedirs('./lab_data/checkpoints/inputs/images', exist_ok=True)

    csd_df = generate_csd_output(min_sol_counts=100, min_habit_counts=1000, save_outputs=True)
    raw_df = pd.read_csv('./lab_data/raw_data/summer_hts_data_matt.csv')

    if args.input == 'image':
        if args.from_scratch or len(os.listdir(f'./csd/checkpoints/inputs/{args.mode}_images')) == 0:
            gen = ImageGenerator(csd_df)

    if args.from_scratch or len(os.listdir(f'./lab_data/checkpoints/inputs/images')) == 0:
        gen = LabImageGenerator(raw_df)


    mix_fastai(set_for_val='lab')




