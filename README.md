# Jeopardy
Analysis and value prediction for jeopardy dataset


Data source: https://www.kaggle.com/tunguz/200000-jeopardy-questions

# Install prerequisities
1. Go to base directory and locate *requirements.txt*
2. Run the command: `pip install -r requirements.txt`

# EDA and clean data.
1. Read and go through `notebooks/eda.ipynb` for feature engineering and transformations.
2. Change directory to `src` by command: `cd src`
3. Clean and transform data, by running script `clean_transform_data.py` with appropriate arguments. Command for running the script `python3 clean_transform_data.py <input_csv_file> <out_csv_file>`
	1. The `Air Date` feature is encoded using binary encoding, with 01/01/2000 as breakpoint. The reason for this is proved in `notebooks/eda.ipynb` 
	2. The text features i.e `Category`, `Question`, `Answer` are cleaned of punctuation and stopwords.
4. Design matrix brief: The design matrix (The final feature matrix) is generated by concatenating encoded `Air Date` and `Round` to the appropriate text vectors.

**After Cleaning the data, We can move onto training the models.**
We tried 3 differnt models. Each improving the error on previous model. 

# Model training
### Linear Regression
Train a baseline linear regression model by following the steps below:
1. Move to `src` directory: `cd src` 
2. Train linear regression: `python3.8 train_linear_regression.py <input_filepath>`
3. We were able to minimize RMSE upto `332.76789076927827` and `806.6202793687579` for training and test data respectively.

Now that we have our baseline. We move onto more complex models. We can tell from the errors reported above that the model is overtrained. We will try to mitigate this in our pursuit of best model.

### Random Forest
Train a Random forest model by following the steps below:
1. Move to `src` directory: `cd src` 
2. Train random forest: `python3.8 train_random_forest.py <input_filepath>`
3. We were able to minimize RMSE upto `526.7433033098621` and `538.565862801748` for training and test data respectively.

As we can see there is definitly improvement on test set from linear regression. The model isn't overtrained, but can we reduce the error further? We will try to finetune the pretrained ***Hugging face*** pretrained tranformers in the next step

### Fine tune bert
Hugging face was throwing an error on my GPU. So I reduced the number of samples in dataset and ran the program on CPU, due to lack of time.
Fine tune bert by following steps:
1. Move to `src` directory: `cd src` 
2. Train random forest: `python3.8 finetune.py <input_filepath> --epochs <num_epochs> --data_frac <fraction of data>`
	1. I used 5 epochs and 0.1 fraction of data(around 20k datapoints).
3. We were able to minimize RMSE upto `106.00519506189207` and `125.75546357125947` for training and test data respectively.

We see a big improvement with bert. However, this is a very big model and requires significant resources to train.