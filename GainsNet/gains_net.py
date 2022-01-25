import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


class GainsNet:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.X_test: np.ndarray = X_test
        self.y_test: np.ndarray = y_test

    def get_lstm_model(self, LR):
        input_dim = self.X_train.shape[1]
        feature_size = self.X_train.shape[2]
        output_dim = self.y_train.shape[1]

        model = Sequential()
        model.add(Bidirectional(LSTM(units=128), input_shape=(
            input_dim, feature_size)))
        model.add(Dense(64))
        model.add(Dense(units=output_dim))
        model.compile(optimizer=Adam(learning_rate=LR), loss='mse')

        return model

    def fit_model(self, model, BATCH_SIZE, N_EPOCH):
        return model.fit(self.X_train, self.y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(self.X_test, self.y_test),
                         verbose=2, shuffle=False)
