import pandas as pd
import os

def get_robot_labelling_df():
    ref = pd.read_csv('./dl_morph_labelling/raw_data/summer_hts_data_matt.csv', index_col=0)
    fnames = []
    labels = []
    api = []
    for drug in ref.api:
        parent_name = f'./dl_morph_labelling/images/raw_images/{drug}'
        parent_content = os.listdir(parent_name)
        for name in parent_content:
            api.append(drug)
            fnames.append(f'{parent_name}/{name}')
            labels.append(ref.eye_morphology[ref['api'] == drug].values[0])
    df = pd.DataFrame({'api': api,
                       'fname': fnames,
                       'label': labels})
    return df