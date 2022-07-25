import argparse
import os

parser = argparse.ArgumentParser(usage='python main.py')
parser.add_argument('--from_scratch', action='store_true', dest='from_scratch')
parser.add_argument('-i', '--input', action='store', dest='input',
                    default='image', choices=['image', 'mordred_descriptor'])
parser.add_argument('-s', '--solvent', action='store', dest='solvent',
                    default='all', choices=['all', 'ethanol', 'methanol', 'water',
                                            'ethyl acetate', 'acetone',
                                            'hexane', 'acetonitrile',
                                            'diethyl ether', 'toluene',
                                            'benzene', 'pentane',
                                            'tetrahydrofuran', 'dimethylsulfoxide',
                                            'isopropanol', 'dimethylformamide',
                                            'cyclohexane', 'heptane', 'best_single']) # Add as needed
parser.add_argument('-j', '--join_mode', action='store', dest='mode',
                    default='concat', choices=['concat', 'one_hot', 'drop'])
parser.add_argument('--no_augs', action='store_true', dest='no_augs', default=False)
parser.add_argument('--robot_test', action='store_true', dest='robot_test', default=False)
parser.add_argument('--gpu_idx', action='store', dest='gpu_idx', default='0',
                  choices=['0', '1', '2', '3', '4', '5'])
parser.add_argument('-m', '--model', action='store', dest='model', default='resnet18',
                  choices=['resnet18', 'convnext_tiny_in22k', 'swinv2_cr_tiny_ns_224',
                           'vit_tiny_patch16_224', 'vit_tiny_patch16_384', 'swinv2_tiny_window16_256',
                           'swinv2_tiny_window8_256', 'convnext_tiny', 'efficientnet_l2'])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
