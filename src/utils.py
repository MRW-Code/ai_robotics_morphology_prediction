import argparse

parser = argparse.ArgumentParser(usage='python main.py')




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
