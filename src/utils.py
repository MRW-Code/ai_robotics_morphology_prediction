import argparse

parser = argparse.ArgumentParser(usage='python main.py')
parser.add_argument('--from_scratch', action='store_true', dest='from_scratch')
parser.add_argument('-i', '--input', action='store', dest='input',
                    default='mordred_descriptor', choices=['image', 'mordred_descriptor', 'rdkit_descriptor',
                                              'mol2vec', 'ecfp', 'pubchem_fp', 'maccs',
                                              'spectrophore', 'weave_graph'])
parser.add_argument('-s', '--solvent', action='store', dest='solvent',
                    default='all', choices=['all','ethanol', 'methanol', 'water']) # Add as needed
parser.add_argument('-j', '--join_mode', action='store', dest='mode',
                    default='concat', choices=['concat', 'one_hot', 'drop'])
parser.add_argument('--no_augs', action='store_true', dest='no_augs', default=False)

# parser.add_argument('-m', '--model', action='store', dest='model',
#                     default='RF', choices=['RF', 'ResNet'])
# parser.add_argument('--kfold', action='store_true', dest='kfold')
# parser.add_argument('-u', '--user', action='store', dest='user',
#                     default='matt', choices=['laura', 'matt'])
# parser.add_argument('--gen_data', action='store_false', dest='load_data')
# parser.add_argument('--deep', action='store_true', dest='is_deep')
# parser.add_argument('--dset', action='store', dest='dset',
#                     default='descriptor', choices=['descriptor', 'image'])
# parser.add_argument('--binary', action='store_false', dest='binary')

args = parser.parse_args()
