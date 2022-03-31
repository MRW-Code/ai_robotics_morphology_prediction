from src.utils import args
from src.inputs import RepresentationGenerator
from csd.parse_csd_files import PreprocessingCSD
from csd.cleaning import do_cleaning
import pandas as pd
import os

if __name__ == '__main__':
    # Parse CSD output files
    if args.from_scratch:
        parser = PreprocessingCSD(save_output=True)
        csd_df = parser.build_csd_output()
        csd_df = do_cleaning(csd_df, min_sol_count=100, min_habit_count=1000, save_output=True)
        input_gen = RepresentationGenerator(csd_df, save_output=True)
        inputs = input_gen.ml_set
    else:
        inputs = pd.read_csv(f'./checkpoints/inputs/{args.input}_dataset.csv', index_col=0)




    print('done')


    # Make sure all other inputs work
    ## add image inputs
    ### add models
    #### add arguemtns to reduce the model complexity e.g single solvents.

