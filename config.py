DATASET_FILEPATH = "Sample.csv" # filepath to dataset: comma-separated file with cells x genes, cell labels in column named as "type"
OUTPUT_FILEPATH = ""            # path to output directory 

K = 5                           # k-fold cross-validation
IDEAL_PORTION_FEATURES = 0.01   # ideal portion of features to be selected: 0-1
POPULATION_SIZE = 30            # population size
ITERATION_NUM = 100             # number of iteration
F = 0.8                         # mutation rate: 0-1
CR = 0.8                        # crossover rate: 0-1

#---------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


print('Reading dataset')
data = pd.read_csv(DATASET_FILEPATH, sep=',', index_col=0)
X = data.drop('type', axis=1)
y = data['type']

splits = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
X_trains = []
X_tests = []
y_trains = []
y_tests = []
for train_index, test_index in splits.split(X, y):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    X_trains.append(X_train)
    X_tests.append(X_test)
    y_trains.append(y_train)
    y_tests.append(y_test)

NUM_FEATURES = X.shape[1] # total number of genes
ELITISM = 2.0             # elitism threshold: 0-2 (set in QDESVM())
