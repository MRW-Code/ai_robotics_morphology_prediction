import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import tqdm
from joblib import Parallel, delayed
import os
import numpy as np
import PIL

class ImageGenerator():

    def __init__(self, csd_output):
        print('Generating API Images')
        self.dataset = self.prep_df(csd_output)
        self.api_smiles = self.dataset['api_smiles']
        self.sol_smiles = self.dataset['sol_smiles']
        self.refcodes = self.dataset['REFCODE']
        self.sol_names = self.dataset['Solvent']
        self.labels = self.dataset['Habit']
        self.gen_all_images()

    def prep_df(self, csd_output):
        df = csd_output.rename(columns={'SMILES' : 'api_smiles'})
        sol_df = pd.read_csv('./csd/raw_data/solvent_smiles.csv')
        join = pd.merge(sol_df, df, on='Solvent').rename(columns={'SMILES': 'sol_smiles'})
        return join

    def smile_to_image(self, api_smile, refcode, sol_smile, sol_name, label):
        try:
            api_mol = Chem.MolFromSmiles(api_smile)
            api_img = np.array(Draw.MolToImage(api_mol, size=[250, 250]))
            sol_mol = Chem.MolFromSmiles(sol_smile)
            sol_img = np.array(Draw.MolToImage(sol_mol, size=[250, 250]))
            stacked = PIL.Image.fromarray(np.vstack([api_img, sol_img]))
            stacked.save(f'./checkpoints/inputs/images/{refcode}_{sol_name}_{label}.png')
        except:
            print(f'{api_smile}, {refcode}, {sol_smile}, {sol_name}, {label}')
        return None

    def gen_all_images(self):
        os.makedirs('./checkpoints/inputs/images', exist_ok=True)
        Parallel(n_jobs=os.cpu_count())\
            (delayed(self.smile_to_image)(i, j, k, l, m) for i, j, k, l, m in tqdm.tqdm(zip(self.api_smiles,
                                                                          self.refcodes,
                                                                          self.sol_smiles,
                                                                          self.sol_names,
                                                                          self.labels), ncols=80))

