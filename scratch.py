import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import timm
import torch


# arry1 = np.array([0.4242105263157895,
#                   0.40315789473684205,
#                   0.4342105263157895,
#                   0.36210526315789476,
#                   0.4131578947368421])
#
# mean = np.mean(arry1)
# print(mean)

# df = pd.read_csv('./checkpoints/inputs/mordred_descriptor_dataset.csv')
# df = pd.read_csv('./csd/processed/raw_csd_output.csv')
#
# x = df.Habit.value_counts()[0:30]
#
# plt.bar(x.index, x.values)
# plt.xlabel('CSD Morphology Label')
# plt.ylabel('Number of Data Points')
# plt.xticks(rotation=90)
# plt.savefig('./test', bbox_inches='tight')
# # plt.show()
#
# print('done')

# cf_matrix = [[int(71), int(11), int(18)],
#              [int(4), int(57), int(9)],
#              [int(16), int(2), int(12)]]
# cf_matrix2 = [[int(71), int(11), int(18)],
#              [int(4), int(57), int(9)],
#              [int(16), int(2), int(12)]]
#
# test = [cf_matrix, cf_matrix2]
#
#
# print(np.sum(test, axis=0))

# total = np.add(cf_matrix, cf_matrix2)
# print(total)

# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
#
# # ax.set_title('ResNet18')
# ax.set_title('ConvNeXt Tiny')
#
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual ');
#
# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['Block','Needle', 'Plate'])
# ax.yaxis.set_ticklabels(['Block','Needle', 'Plate'])
#
# ## Display the visualization of the Confusion Matrix.
# # plt.savefig('/home/matthew/Documents/particle_flow_logs/confmat_resnet.png')
# # plt.savefig('/home/matthew/Documents/particle_flow_logs/confmat_convnext.png')
#
# plt.show()

avail_pretrained_models = timm.list_models(pretrained=True)