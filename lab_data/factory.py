import pandas as pd
import re
from tqdm import tqdm
import os

def get_lab_aug_df():
    print('GETTING LAB AUG DF')
    image_dir = f'./lab_data/checkpoints/inputs/aug_images'
    paths = [f'{image_dir}/{x}' for x in tqdm(os.listdir(image_dir))]
    labels = [re.findall(r'.*_(.*).png', y)[0] for y in tqdm(paths)]
    model_df = pd.DataFrame({'fname': paths,
                             'label': labels})
    model_df['is_valid'] = 0
    return model_df