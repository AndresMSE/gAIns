import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import tensorflow as tf


class GainsPlottter:
    @staticmethod
    def plot_stock_timeline(title: str, dates, y):
        _, ax = plt.subplots(figsize=(10, 3))
        ax.plot(dates, y, label=f'{title} stock')
        ax.set(xlabel="Date",
               ylabel="Value",
               title=title)
        date_form = DateFormatter("%Y")
        ax.xaxis.set_major_formatter(date_form)
        ax.plot()

    @staticmethod
    def plot_train_results(history: tf.keras.callbacks.History):
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_predict_result(title, predict_result, real_price):
        predict_result['predicted_mean'] = predict_result.mean(axis=1)
        real_price['real_mean'] = real_price.mean(axis=1)

        plt.figure(figsize=(16, 8))
        plt.plot(real_price["real_mean"])
        #plt.plot(predict_result["predicted_mean"], color='r')
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("Real price", "Predicted price"),
                   loc="upper left", fontsize=16)
        plt.title(title, fontsize=20)
        plt.show()
