import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def prediction(dataset, kolom):
    loaded_model = joblib.load('outputModel/modelNaiveBayesJl.pkl')
    vect = TfidfVectorizer(max_features=5600)
    data = vect.fit_transform(dataset[kolom])
    predictions = loaded_model.predict(data)

    dos = []

    for prediction in predictions:
        if prediction == "NETRAL":
            do = 0
        elif prediction == "POSITIF":
            do = 1
        else:
            do = 2
        dos.append(do)            
    return dos


if __name__ == "__main__":
    df = pd.read_csv('dataset/data_clean150.csv')
    klm = 'stemmed'
    hasilPrediksi = prediction(df, klm)
    print(hasilPrediksi[1])
    # for i in hasilPrediksi:
    #     print(i)
