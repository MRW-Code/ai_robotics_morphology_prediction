import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# arry1 = np.array([0.4242105263157895,
#                   0.40315789473684205,
#                   0.4342105263157895,
#                   0.36210526315789476,
#                   0.4131578947368421])
#
# mean = np.mean(arry1)
# print(mean)

# df = pd.read_csv('./checkpoints/inputs/mordred_descriptor_dataset.csv')
df = pd.read_csv('./csd/processed/raw_csd_output.csv')

x = df.Habit.value_counts()[0:30]

plt.bar(x.index, x.values)
plt.xlabel('CSD Morphology Label')
plt.ylabel('Number of Data Points')
plt.xticks(rotation=90)
plt.savefig('./test', bbox_inches='tight')
# plt.show()

print('done')