from aaia.helpers import save_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

# Params
HIDDEN_SIZE = 64


def create_dataset(df, target_columns, lookback):
    dataset = df[target_columns].astype('float32').to_numpy()

    X, y = [], []
    for i in range(len(dataset) - lookback):
        features = dataset[i:i + lookback]
        targets = dataset[i + 1:i + lookback + 1]
        X.append(torch.tensor(features))
        y.append(torch.tensor(targets))
    return torch.stack(X), torch.stack(y)


class TimeSeriesModel(nn.Module):

    def __init__(self, target_columns):
        super().__init__()
        size = len(target_columns)
        self.lstm = nn.LSTM(input_size=size,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=1,
                            batch_first=True)
        self.linear = nn.Linear(HIDDEN_SIZE, size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


def train(df_train,
          df_test,
          target_columns,
          lookback,
          n_epochs,
          batch_size,
          output_path=None):
    print('==== Time series training ====')

    X_train, y_train = create_dataset(df=df_train,
                                      target_columns=target_columns,
                                      lookback=lookback)
    X_test, y_test = create_dataset(df=df_test,
                                    target_columns=target_columns,
                                    lookback=lookback)

    model = TimeSeriesModel(target_columns)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                             shuffle=True,
                             batch_size=batch_size)

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # validation (info only, not influencing training)
        if epoch % 10 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print('Epoch %d: train RMSE %.4f, test RMSE %.4f' %
              (epoch, train_rmse, test_rmse))

    if output_path:
        save_model(model, output_path)

    return model


def predict(model, df, target_columns, lookback):
    X, y = create_dataset(df=df,
                          target_columns=target_columns,
                          lookback=lookback)
    return model(X)[:, -1, :].detach().numpy()


def plot(df_train,
         df_test,
         model,
         target_columns,
         lookback,
         y_max,
         title,
         x_label,
         output_paths=None):
    X_train, y_train = create_dataset(df=df_train,
                                      target_columns=target_columns,
                                      lookback=lookback)
    X_test, y_test = create_dataset(df=df_test,
                                    target_columns=target_columns,
                                    lookback=lookback)

    n_train = df_train.shape[0]
    n = n_train + df_test.shape[0]
    X = np.concatenate([df_train[target_columns].astype('float32').to_numpy(),
                        df_test[target_columns].astype('float32').to_numpy()])

    with torch.no_grad():
        predictions_train = predict(model=model,
                                    df=df_train,
                                    target_columns=target_columns,
                                    lookback=lookback)
        predictions_test = predict(model=model,
                                   df=df_test,
                                   target_columns=target_columns,
                                   lookback=lookback)

    for i, target_column in enumerate(target_columns):
        # shift train predictions for plotting
        train_plot = np.ones(n) * np.nan
        train_plot[lookback:n_train] = predictions_train[:, i]
        # shift test predictions for plotting
        test_plot = np.ones(n) * np.nan
        test_plot[n_train + lookback:n] = predictions_test[:, i]

        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(target_column)
        plt.title(title)
        plt.ylim((0, y_max[i]))
        plt.plot(X[:, i], color='tab:blue', label='real data')
        plt.plot(train_plot, color='tab:orange',
                 label='prediction on training data')
        plt.plot(test_plot, color='tab:green', label='prediction on test data')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

        if output_paths:
            plt.savefig(output_paths[i], bbox_inches='tight')
        else:
            plt.show()
