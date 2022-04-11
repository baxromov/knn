import pandas as pd
import warnings

from utils import get_euclidean, get_nearest, get_dominant_class

warnings.filterwarnings('ignore')

file_columns = ['class', 'sepal_len', 'sepal_width', 'petal_len', 'petal_width']

data = pd.read_csv('iris_tbl.csv', header=None, names=file_columns)

# To prevent bias in learning, we shuffle the data before dividing it into development and testing set.
data = data.sample(frac=1).reset_index(drop=True)
data['seq'] = data.index

"""
Divide into Development and Test set
Dataset is split into development and test set in the ratio 75:25 respectively.
"""

dev_size = int(data.shape[0] * 0.75)
test_size = int(data.shape[0] * 0.25)
# Take first 75% of the data as dev set
dev = data[:dev_size]
# Take last 25% of the data as test set
test = data[test_size:]
dev2 = dev.values

"""
 1. Calculate Euclidean distance
"""
test['seq'] = test.index
test2 = test.values
test_eud = []
l = len(test)
for i in range(l):
    test_eu_distance = []
    for j in range(len(dev)):
        index = dev2[j][5]
        ed = get_euclidean(test2[i][:-2], dev2[j][:-2])
        test_eu_distance.append((ed, index))

    test_eu_distance.sort(key=lambda x: x[0])
    test_eu_distance = [i[1] for i in test_eu_distance]
    test_eud.append(test_eu_distance)

test['euclidean'] = test_eud

"""
2. Pick Nearest Neighbors
"""
test['eu'] = test.apply(lambda x: get_nearest(x, 'euclidean', 3), axis=1)
print(test[file_columns + ['eu']].head())

"""
3. Classify the test set
Pick the most dominant class among 3 nearest neighbors
"""
test['eu_class'] = test['eu'].apply(lambda row: get_dominant_class(dev, row))
print(test[file_columns + ['eu_class']].head())

"""
4. Accuracy over test set
"""
test_acc = test[test['class'] == test['eu_class']].shape[0] / test.shape[0]
print('Test Accuracy: {:2.4f}%'.format(test_acc * 100))
