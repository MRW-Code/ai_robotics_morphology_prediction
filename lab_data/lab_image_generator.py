from src.utils import args
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import tqdm
from joblib import Parallel, delayed
import os
import numpy as np
import PIL

class LabImageGenerator():

    def __init__(self, raw_df):
        print('Generating API Images')
        self.dataset = raw_df
        self.api_smiles = self.dataset['SMILES']
        self.api_names = self.dataset['api']
        self.labels = self.dataset['eye_morphology']
        self.gen_all_images()


    def smile_to_image(self, api_smile, api_name, label):
        try:
            api_mol = Chem.MolFromSmiles(api_smile)
            api_img = np.array(Draw.MolToImage(api_mol, size=[250, 250]))
            api_img = PIL.Image.fromarray(api_img)
            api_img.save(f'./lab_data/checkpoints/inputs/images/{api_name}_{label}.png')
        except:
            print(f'{api_smile}, {refcode}, {sol_smile}, {sol_name}, {label}')
        return None

    def gen_all_images(self):
        Parallel(n_jobs=os.cpu_count())\
            (delayed(self.smile_to_image)(i, j, k) for i, j, k in tqdm.tqdm(zip(self.api_smiles,
                                                                          self.api_names,
                                                                          self.labels), ncols=80))

