from dl_morph_labelling.preprocessing import get_robot_labelling_df
from dl_morph_labelling.model import robot_kfold_fastai, get_robot_external_set
import pandas as pd
import os

if __name__ == '__main__':
    os.makedirs('./dl_morph_labelling/checkpoints', exist_ok=True)
    robot_df = get_robot_labelling_df()
    # robot_kfold_fastai(robot_df, n_splits=5)
    get_robot_external_set(robot_df, 1)
