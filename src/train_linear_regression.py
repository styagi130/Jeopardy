import argparse
import pathlib

import pandas as pd
import numpy as np
from scipy import sparse

from sklearn import linear_model, metrics, model_selection, feature_extraction


def concat_feats(text_vectorizer, df):
    text_feats = text_vectorizer.transform(df.loc[:, "text_feats"].values)

    numeric_feats = np.hstack((
        df.loc[:, " Air Date"].values.reshape(-1, 1),
        df.loc[:, " Round"].values.reshape(-1, 1)
        )
    )
    return sparse.hstack((text_feats, numeric_feats))

def main(args):
    df = pd.read_csv(args.input_file)
    df.dropna(inplace=True)

    ## Generate_text_feats
    df.loc[:, "text_feats"] = df.loc[:, " Question"] +" "+df.loc[:, " Answer"] +" "+ df.loc[:, " Category"]

    ## Generate train, validation pair
    df_train, df_val = model_selection.train_test_split(df, train_size=0.7)
    
    ### Train tfidf featurizer
    vectorizer = feature_extraction.text.TfidfVectorizer()
    vectorizer.fit(df_train.loc[:, "text_feats"])

    ### Generate full features
    X_train = concat_feats(vectorizer, df_train)
    y_train = df_train.loc[:, " Value"]
    X_val = concat_feats(vectorizer, df_val)
    y_val = df_val.loc[:, " Value"]

    ### Train regressor
    print("~~~~~~~~~~~~~~~~~~~Training Linear Regression~~~~~~~~~~~~~~~~~~~")
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    ### Predictions
    y_pred = lr.predict(X_train)
    rmse_train = metrics.mean_squared_error(y_train, y_pred, squared=False)
    print( f"Achieved RMSE of {rmse_train} on training data")
    y_pred = lr.predict(X_val)
    rmse_val = metrics.mean_squared_error(y_val, y_pred, squared=False)
    print( f"Achieved RMSE of {rmse_val} on validation data")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Use this script to train logistic regression model. Example: python3.8 train_linear_regression.py <input_filepath>")
    parser.add_argument("input_file", metavar="fi", type=pathlib.Path, help="Input filepath, With encoded features.")
    args = parser.parse_args()

    main(args)