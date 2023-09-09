import pandas as pd

TRAIN_DATASET_SIZE = 5000

if __name__ == '__main__':
  df = pd.read_csv('krkopt.data', header=None)

  shuffled = df.sample(frac=1).reset_index(drop=True)

  train_data = shuffled[:TRAIN_DATASET_SIZE]
  test_data = shuffled[TRAIN_DATASET_SIZE:]

  train_data.to_csv('krkopt_train.data', header=False, index=False)
  test_data.to_csv('krkopt_test.data', header=False, index=False)
