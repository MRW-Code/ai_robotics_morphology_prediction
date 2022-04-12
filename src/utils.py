import argparse
import os

parser = argparse.ArgumentParser(usage='python main.py')
parser.add_argument('--from_scratch', action='store_true', dest='from_scratch')
parser.add_argument('-i', '--input', action='store', dest='input',
                    default='mordred_descriptor', choices=['image', 'mordred_descriptor', 'rdkit_descriptor',
                                              'mol2vec', 'ecfp', 'pubchem_fp', 'maccs',
                                              'spectrophore', 'weave_graph'])
parser.add_argument('-s', '--solvent', action='store', dest='solvent',
                    default='all', choices=['all', 'ethanol', 'methanol', 'raw_images',
                                            'ethyl acetate', 'acetone',
                                            'hexane', 'acetonitrile',
                                            'diethyl ether', 'toluene',
                                            'benzene', 'pentane',
                                            'tetrahydrofuran', 'dimethylsulfoxide',
                                            'isopropanol', 'dimethylformamide',
                                            'cyclohexane', 'heptane']) # Add as needed
parser.add_argument('-j', '--join_mode', action='store', dest='mode',
                    default='concat', choices=['concat', 'one_hot', 'drop'])
parser.add_argument('--no_augs', action='store_true', dest='no_augs', default=False)
parser.add_argument('--gpu_idx', action='store', dest='gpu_idx', default='0',
                  choices=['0', '1', '2', '3', '4', '5'])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
