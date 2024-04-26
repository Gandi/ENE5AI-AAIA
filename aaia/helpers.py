import pandas as pd
import torch


def save_dataframe(df, filepath, sep='\t'):
    df.to_csv(filepath, sep=sep, index=False)


def load_dataframe(filepath, sep='\t', header=0):
    df = pd.read_csv(filepath,
                     sep=sep,
                     header=header)
    return df


def split_train_test_df(df, test_coef):
    n = df.shape[0]
    train_size = int(n * (1 - test_coef))
    df_train = df.head(train_size)
    df_test = df.tail(n - train_size)
    return df_train, df_test


def save_model(model, filepath):
    if not filepath.endswith('.pth'):
        filepath += '.pth'
    torch.save(model.state_dict(), filepath)


def load_model(ModelClass, filepath, model_params=None):
    model = ModelClass(model_params)
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    model.eval()
    return model
