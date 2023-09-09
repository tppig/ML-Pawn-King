import numpy as np
import pandas as pd
from libsvm.svm import *
from libsvm.svmutil import svm_train, svm_save_model


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
  df['File_A'] = df['File_A'].map(lambda x: ord(x) - ord('a'))
  df['File_B'] = df['File_B'].map(lambda x: ord(x) - ord('a'))
  df['File_C'] = df['File_C'].map(lambda x: ord(x) - ord('a'))
  df['odw'] = df['odw'].map(lambda x: 1 if x == 'draw' else 0)

  normalized = pd.DataFrame({'odw': df['odw']})

  df = df.drop(columns=['odw'])
  df = (df - df.mean()) / df.std()

  normalized = pd.concat([df, normalized], axis=1)
  return normalized


def search_optimal_hyper_param(df: pd.DataFrame, C_list: list, Gamma_list: list) -> (float, float, float):
  Y = df['odw'].tolist()

  df = df.drop(columns=['odw'])
  X = df.values.tolist()

  prob = svm_problem(Y, X)
  max_r = 0
  hyper_C = None
  hyper_Gamma = None
  for C in C_list:
    for Gamma in Gamma_list:
      param = svm_parameter(f'-t 2 -v 5 -c {2 ** C} -g {2 ** Gamma} -h 0')
      m = svm_train(prob, param)
      if m > max_r:
        max_r = m
        hyper_C = C
        hyper_Gamma = Gamma

  return max_r, hyper_C, hyper_Gamma


def train_final_model(df: pd.DataFrame, C: float, Gamma: float, model_file_name: str) -> None:
  Y = df['odw'].tolist()

  df = df.drop(columns=['odw'])
  X = df.values.tolist()

  prob = svm_problem(Y, X)
  param = svm_parameter(f'-t 2 -c {2 ** C} -g {2 ** Gamma} -h 0')
  model = svm_train(prob, param)
  svm_save_model(model_file_name, model)


if __name__ == '__main__':
  df = pd.read_csv('krkopt_train.data', header=None)
  df.rename(columns={0: 'File_A', 1: 'Rank_A', 2: 'File_B', 3: 'Rank_B', 4: 'File_C', 5: 'Rank_C', 6: 'odw'},
            inplace=True)
  normalized_df = normalize_data(df)

  # Find optimal hyper parameters - coarse-grained
  C_list = list(range(-5, 16, 1))
  Gamma_list = list(range(-15, 4, 1))
  max_rate, C, Gamma = search_optimal_hyper_param(normalized_df, C_list, Gamma_list)

  # Find optimal hyper parameters - fine-grained
  n = 10
  C_low = 0.5 * (max(-5, C - 1) + C)
  C_upper = 0.5 * (min(15, C + 1) + C)
  C_list = list(np.arange(C_low, C_upper, (C_upper - C_low) / n))
  Gamma_low = 0.5 * (max(-15, Gamma - 1) + Gamma)
  Gamma_upper = 0.5 * (min(4, Gamma + 1) + Gamma) + 0.001
  Gamma_list = list(np.arange(Gamma_low, Gamma_upper, (Gamma_upper - Gamma_low) / n))
  max_rate, C, Gamma = search_optimal_hyper_param(normalized_df, C_list, Gamma_list)
  print(f'Optimal hyper parameters: C={C}, Gamma={Gamma}, max_rate={max_rate}')

  # Train final model
  train_final_model(normalized_df, C, Gamma, 'krkopt.model')
