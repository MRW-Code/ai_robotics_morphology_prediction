import os
from tqdm import tqdm
import pandas as pd
import re

class PreprocessingCSD:
    def __init__(self, raw_data_dir='./csd/raw_data'):
        self.raw_dir = raw_data_dir
        self.txt_path, self.smi_path = self.find_csd_output_files()

    def find_csd_output_files(self):
        for file in os.listdir(self.raw_dir):
            if '.txt' in file:
                csd_txt_path = f'{self.raw_dir}/{file}'
            elif '.smi' in file:
                csd_smi_path = f'{self.raw_dir}/{file}'
        return csd_txt_path, csd_smi_path


    @staticmethod
    def remove_salts(file_path):
        """
        Function to remove any SMILE string that represents a salt.
        Looks for any SMILES with a full stop in it.

        @param csd_smi: path to the smi that comes out the CSD
        @return: pd.DataFrame containing SMILES and REFCODES
        """
        with open(file_path, "r") as readfile:
            file_contents = readfile.readlines()
        filtered_contents = [None] * len(file_contents)
        for i, line in enumerate(file_contents):
            if "." not in line:
                filtered_contents[i] = line
        return filtered_contents


    def smi_to_df(self):
        print('PARSING CSD .SMI FILE')
        # Read file with salts removed
        lines = self.remove_salts(self.smi_path)

        # Generate empty df to hold data
        col_headers = ['REFCODE', 'SMILES']
        smiles_dataframe = pd.DataFrame(columns=col_headers)

        # Loop all lines in the file and add variables to df where they should go
        counter = 1
        for line in tqdm(lines, nrows=80):
            if line != None:
                S = line.split('\t')
                R = S[1].replace('\n', '')
                data = pd.Series({'REFCODE': R, 'SMILES': S[0]})
                smiles_dataframe = smiles_dataframe.append(data, ignore_index=True)
            counter += 1
        unique_smiles_dataframe = smiles_dataframe.drop_duplicates(subset=['SMILES'],
                                                                   keep='last')

        return unique_smiles_dataframe


    def extract_info(self, regex, data):
        """
        Applies a given regex string to data and returns the matches.
        @param regex: RegEx string.
        @param data: Data to parse.
        @return: RegEx matches.
        """
        reg = re.compile(regex)
        matches = reg.findall(data)

        if len(matches) > 0:
            return matches[0]
        else:
            return ''

    def parse_data(self, data):
        """
        Parses given data to extract the information needed for each variable.
        @param data: Data to parse.
        @return: All the parsed data.
        """
        REFCODE = self.extract_info(r'(?:REFCODE:\s*([\S|\d].*))', data)
        # Name = self.extract_info(r'(?:Name:\s*([\S|\d].*))', data)
        Habit = self.extract_info(r'(?:Habit:\s*([\S|\d].*))', data)
        From = self.extract_info(r'(?:From:\s*([\S|\d].*))', data)

        return pd.Series({
            'REFCODE': REFCODE,
            'Habit': Habit,
            'Solvent': From
        })

    def txt_to_df(self):
        print('PARSING CSD .txt FILE')
        with open(self.txt_path) as f:
            data = f.read()
            col_headers = ['REFCODE', 'Habit', 'Solvent']
            info = pd.DataFrame(columns=col_headers)
            reg = re.compile(r'REFCODE')
            matches = list(reg.finditer(data))
            for i in range(len(matches)):
                print('\r\t{}\t/\t{}\t\t\t'.format(i, len(matches)), end='')
                m_start = matches[i].start()
                m_end = len(data)
                if i < len(matches) - 1:
                    m_end = matches[i + 1].start()
                refined_data = self.parse_data(data[m_start:m_end])
                info = info.append(refined_data, ignore_index=True)
            return info

    def build_csd_output(self):
        smiles_no_salt = self.smi_to_df()
        raw_csd_output = self.txt_to_df()
        full_CSD_output = pd.merge(raw_csd_output, smiles_no_salt, on='REFCODE', how='inner')
        return full_CSD_output

    # def csd_files_to_df(self):
    #     dir_list = os.listdir(self.raw_dir)
    #     for file in list:
    #         if '.txt' in file:
    #             txt_df =
    #     return None
