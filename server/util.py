import json
import pickle
import numpy as np
import math

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1

    return __model.predict([x])[0]


def get_location_names():
    return __locations


def load_saved_artifacts():
    print("load saved artifacts.....start")
    global __data_columns
    global __locations

    with open("E:/7th Sem Project\BangloreHomePrice\server/artifacts\BHPcolumns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("E:/7th Sem Project\BangloreHomePrice\server/artifacts/bangloreHomePriceModel.pickle", "rb") as f:
        __model = pickle.load(f)
        print("load saved artifacts.... done")

    return __locations


if __name__ == "__main__":
    load_saved_artifacts()
    # print(get_location_names())
    # print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
