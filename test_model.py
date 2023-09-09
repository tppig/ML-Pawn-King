from libsvm.svm import *
from libsvm.svmutil import svm_load_model, svm_predict

from optimal_hyper_param import normalize_data
import pandas as pd

if __name__ == '__main__':
  df = pd.read_csv('krkopt_test.data', header=None)
  df.rename(columns={0: 'File_A', 1: 'Rank_A', 2: 'File_B', 3: 'Rank_B', 4: 'File_C', 5: 'Rank_C', 6: 'odw'},
            inplace=True)
  df = normalize_data(df)

  model = svm_load_model('krkopt.model')

  Y = df['odw'].tolist()
  df = df.drop(columns=['odw'])
  X = df.values.tolist()
  _, p_acc, _ = svm_predict(Y, X, model)
