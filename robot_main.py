from dl_morph_labelling.preprocessing import get_robot_labelling_df
from dl_morph_labelling.model import robot_kfold_fastai, robot_external_test
import pandas as pd
import os
from src.utils import args
if __name__ == '__main__':
    os.makedirs('./dl_morph_labelling/checkpoints', exist_ok=True)
    robot_df = get_robot_labelling_df()

    if args.robot_test:
        robot_external_test(robot_df)
    else:
        robot_kfold_fastai(robot_df, n_splits=10)
