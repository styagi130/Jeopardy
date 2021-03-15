import argparse
import pathlib
import re
import string

import pandas as pd
import numpy as np
from sklearn import preprocessing


stop_words_path = pathlib.Path("./../inputs/stopwords.txt")

remove_punctuation = lambda text: text.translate(str.maketrans('', '', string.punctuation))

def main(args):
    df = pd.read_csv(args.input_file)
    stop_words = stop_words_path.read_text().split("\n")

    ### Impute na values
    df.fillna(" ", inplace=True)

    ### Clean Questions, Answers and Categories
    ## Remove punctuations
    df.loc[:, " Question"] = df.loc[:, " Question"].apply(remove_punctuation)
    df.loc[:, " Answer"] = df.loc[:, " Answer"].apply(remove_punctuation)
    df.loc[:, " Category"] = df.loc[:, " Category"].apply(remove_punctuation)
    ## Remove stopwords
    df.loc[:, " Question"] = df.loc[:, " Question"].apply(lambda sent: " ".join([word for word in sent.split() if word not in stop_words]))
    df.loc[:, " Answer"] = df.loc[:, " Answer"].apply(lambda sent: " ".join([word for word in sent.split() if word not in stop_words]))
    df.loc[:, " Category"] = df.loc[:, " Category"].apply(lambda sent: " ".join([word for word in sent.split() if word not in stop_words]))

    ### Clean and convert value to integers
    df.loc[:, " Value"] = df.loc[:, " Value"].apply(lambda value: re.sub("[,$]", "", value) if value != "None" else 0).astype(int)

    ### Encode datetime as explained in EDA notebook
    date_threshold = np.datetime64("2000-01-01T00:00:00.000000000")
    df.loc[:, " Air Date"] = pd.to_datetime(df.loc[:, " Air Date"].values)
    df.loc[:, " Air Date"] = df.loc[:, " Air Date"].apply(lambda x: 1 if x>date_threshold else 0)

    ### Encode Rounds as explained in EDA notebook
    le = preprocessing.LabelEncoder()
    df.loc[:, " Round"] = le.fit_transform(df.loc[:, " Round"].values)

    df.fillna(" ", inplace=True)
    ### Persist to disc
    print(df.isna().sum())
    df.to_csv(args.output_file, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Use this script to remove stop words, clean and transform data!! Example: python3 clean_transform_data.py <input_csv_file> <out_csv_file>")
    parser.add_argument("input_file", metavar="fi", type=pathlib.Path, help="Input csv file path")
    parser.add_argument("output_file", metavar="fo", type=pathlib.Path, help="Output csv file path")
    args = parser.parse_args()

    main(args)
