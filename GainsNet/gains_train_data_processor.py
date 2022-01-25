import numpy as np


class GainsTrainDataProcessor:
    def __init__(self) -> None:
        self.n_steps_in = 3
        self.n_steps_out = 1

    def get_X_y(self, X_data, y_data):
        X = list()
        y = list()
        yc = list()

        length = len(X_data)
        for i in range(0, length, 1):
            X_value = X_data[i: i + self.n_steps_in][:, :]
            y_value = y_data[i + self.n_steps_in: i +
                             (self.n_steps_in + self.n_steps_out)][:, 0]
            yc_value = y_data[i: i + self.n_steps_in][:, :]
            if len(X_value) == 3 and len(y_value) == 1:
                X.append(X_value)
                y.append(y_value)
                yc.append(yc_value)

        return np.array(X), np.array(y), np.array(yc)

    def get_train_test_predict_index(self, dataset, X_train):

        # get the predict data (remove the in_steps days)
        train_predict_index = dataset.iloc[self.n_steps_in: X_train.shape[0] +
                                           self.n_steps_in + self.n_steps_out - 1, :].index
        test_predict_index = dataset.iloc[X_train.shape[0] +
                                          self.n_steps_in:, :].index

        return train_predict_index, test_predict_index

    def split_train_test(self, data):
        train_size = round(len(data) * 0.7)
        data_train = data[0:train_size]
        data_test = data[train_size:]
        return data_train, data_test
