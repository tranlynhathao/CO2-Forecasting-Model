import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def create_ts_data(data, window_size=10):
    count = 1
    while count < window_size:
        data["co2_{}".format(count)] = data["co2"].shift(-count)
        count += 1
    data["target"] = data["co2"].shift(-count)
    data = data.dropna(axis=0)
    return data


def main():
    data = pd.read_csv("co2.csv")
    data["time"] = pd.to_datetime(data["time"])
    data["co2"] = data["co2"].interpolate()

    # ax.plot(data["time"], data["co2"])
    # ax.set_xlabel("Time")
    # ax.set_ylabel("CO2")
    # plt.show()

    window_size = 5
    train_ratio = 0.8
    data = create_ts_data(data, window_size)
    num_samples = len(data)
    x = data.drop(["time", "target"], axis=1)
    y = data["target"]
    x_train = x[:int(num_samples*train_ratio)]
    y_train = y[:int(num_samples*train_ratio)]
    x_test = x[int(num_samples * train_ratio):]
    y_test = y[int(num_samples * train_ratio):]

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    # y_predict = reg.predict(x_test)
    # mae = mean_absolute_error(y_test, y_predict)
    # mse = mean_squared_error(y_test, y_predict)
    # r2 = r2_score(y_test, y_predict)
    #
    # print("Mean absolute error: {}".format(mae))
    # print("Mean squared error: {}".format(mse))
    # print("R2: {}".format(r2))
    #
    # fig, ax = plt.subplots()
    # ax.plot(data["time"][:int(num_samples*train_ratio)], y_train, label="train")
    # ax.plot(data["time"][int(num_samples*train_ratio):], y_test, label="test")
    # ax.plot(data["time"][int(num_samples*train_ratio):], y_predict, label="prediction")
    # ax.set_xlabel("Time")
    # ax.set_ylabel("CO2")
    # ax.legend()
    # ax.grid()
    # plt.show()

    current_data = [380.5, 390, 390.2, 394, 394.4]
    # prediction = reg.predict([current_data]).tolist()
    for i in range(10):
        print(current_data)
        prediction = reg.predict([current_data]).tolist()
        print("CO2 in week {} is {}".format(i+1, prediction[0]))
        current_data = current_data[1:] + prediction
        print("----------------------")




if __name__ == '__main__':
    main()
