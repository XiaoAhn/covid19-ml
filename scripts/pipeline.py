from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,mean_squared_error,mean_absolute_error
from joblib import dump
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import datetime
import os
import pickle

def get_numeric_correlations(df, method='pearson'):
    '''
    Prints a table of pairwise correlation for all numeric
    columns in a pandas DataFrame object. Returns null. 
    '''
    # Check pandas df
    if isinstance(df, pd.DataFrame):
        numeric_df = df.select_dtypes(include=['number'])
        print(numeric_df.corr(method=method))
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def get_numeric_histograms(df, nbins=40, log=False):
    '''
    Displays histograms of all numeric columns in a 
    pandas DataFrame. Returns null.
    '''
    # Check pandas df
    if isinstance(df, pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['number'])
        for col in numeric_cols.columns:

            # Remove NaN values
            toplot = numeric_cols[col].dropna()

            # Make histogram
            plt.figure(figsize=(10,5))
            plt.hist(toplot, bins=nbins, log=log)
            plt.title(f"Histogram of {col}", fontsize=15)
            plt.xlabel(f"Value for {col}", fontsize=10)
            plt.ylabel("Number of Observations", fontsize=10)
            plt.yticks(fontsize=9)
            plt.xticks(fontsize=9)
            plt.show()
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def get_train_test(df, test_size=0.1, random_state=1, time_series=False, 
                   validation=False, validation_low=0.8, validation_high=0.9):
    '''
    Calls sklearn's train_test_split on an input df and returns 
    the resulting train and test set DataFrames. Takes in parameters
    for training size and random seed. Ignores the potential need
    for a validation set.
    '''

    # Check pandas df
    if isinstance(df, pd.DataFrame):

        # Non-time series split
        if not time_series:
            train, test = train_test_split(df, train_size=1-test_size, 
                                           test_size=test_size, 
                                           random_state=random_state)
            return (train, test)
        elif time_series:
            df.loc[:,'date'] = df['date'].astype('datetime64')
            if not validation:
                testcutoff = df['date'].quantile(1-test_size)
                train = df[df['date'] <= testcutoff]
                test  = df[df['date'] > testcutoff]
                return (train, test)
            elif validation:
                validationlow = df['date'].quantile(validation_low)
                validationhigh = df['date'].quantile(validation_high)
                testcutoff = df['date'].quantile(1-test_size)
                train = df[df['date'] <= validationlow]
                validation = df[(df['date'] > validationlow) & (df['date'] <= validationhigh)]
                test  = df[df['date'] > testcutoff]
                return (train, validation, test)
                    
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def impute_missing(train, test, validation=None, how='median', verbose=False):
    '''
    Imputes missing values for numeric columns. Returns
    train and test data with missing values imputed.
    (Imputation done based on train columns)
    Parameter:
    df - pandas DataFrame to impute
    how - how to impute. default median. also accepts
     'mean' or some numeric value.
    '''
    # Check pandas df
    if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):
        numeric_cols = train.select_dtypes(include=['number']).columns

        # Loop over numeric columns
        for col in numeric_cols:

            # Get missing value counts
            trainnas = train[col].isna().sum()
            testnas = test[col].isna().sum()
            if validation is not None:
                validationnas = validation[col].isna().sum()
            else:
                validationnas = 0

            # Only impute if no missing values (saves time)
            if (trainnas > 0) or (testnas > 0) or (validationnas > 0):

                # Verbosity
                if verbose == True:
                    print(f"Imputing {trainnas+testnas:,} missing values for {col}")

                # Impute
                if how=='median':
                    train.loc[:,col] = train.loc[:,col].fillna(train.loc[:,col].median())
                    test.loc[:,col] = test.loc[:,col].fillna(train.loc[:,col].median())
                    if validation is not None:
                        validation.loc[:,col] = validation.loc[:,col].fillna(validation.loc[:,col].median())
                elif how=='mean':
                    train.loc[:,col] = train.loc[:,col].fillna(train.loc[:,col].mean())
                    test.loc[:,col] = test.loc[:,col].fillna(train.loc[:,col].mean())
                    if validation is not None:
                        validation.loc[:,col] = validation.loc[:,col].fillna(validation.loc[:,col].mean())
                elif isinstance(how,(int,float)):
                    train.loc[:,col] = train.loc[:,col].fillna(how)
                    test.loc[:,col] = test.loc[:,col].fillna(how)
                    if validation is not None:
                        validation.loc[:,col] = validation.loc[:,col].fillna(how)
                else:
                    raise TypeError("Please pass a valid 'how' value.")
            
            else:
                if verbose == True:
                    print(f"No missing values for {col}")

        # Return with or without validation
        if validation is not None:
            return train,validation,test
        else:
            return train,test
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def normalize_vars(train, test, validation=None, verbose=False):
    '''
    Normalizes numeric variables in train and test dataset to 
    have mean 0 and standard deviation of 1. Returns the train and
    test dataset with normalized features.
    '''
    # Check pandas df
    if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):            

        # Split into numeric and non-numeric
        train_numeric = train.select_dtypes(include=['number'])
        test_numeric = test.select_dtypes(include=['number'])
        train_nonnumeric = train.select_dtypes(exclude=['number'])
        test_nonnumeric = test.select_dtypes(exclude=['number'])
        if validation is not None:
            validation_numeric = validation.select_dtypes(include=['number'])
            validation_nonnumeric = validation.select_dtypes(exclude=['number'])

        # Verbosity
        if verbose == True:
            print(f"The following columns will be normalized: {list(train_numeric.columns)}")

        # Scale numeric data
        scaler = StandardScaler()
        trainscale = pd.DataFrame(scaler.fit_transform(train_numeric))
        testscale = pd.DataFrame(scaler.transform(test_numeric))
        if validation is not None:
            validationscale = pd.DataFrame(scaler.transform(validation_numeric))

        # Fix columns and indices
        trainscale.columns = train_numeric.columns
        trainscale.index = train_numeric.index
        testscale.columns = test_numeric.columns
        testscale.index = test_numeric.index
        if validation is not None:
            validationscale.columns = validation_numeric.columns
            validationscale.index = validation_numeric.index

        # Merge back onto non-numeric data
        train = trainscale.join(train_nonnumeric)
        test = testscale.join(test_nonnumeric)
        if validation is not None:
            validation = validationscale.join(validation_nonnumeric)
            return (train, validation, test)
        else:
            return (train, test)
    else: 
        raise TypeError("df must be a pandas DataFrame object")


def build_regressors(models, params_grid, 
                      train_features, train_outcome, 
                      test_features, test_outcome,
                      save_path, validation_low, validation_high):
    '''
    Trains a number of models and returns a DataFrame 
    of these models and their resulting evaluation metrics.
    Parameters:
     models - dictionary of sklearn models to fit
     parameters - dictionary of parameters to test for each of above models
     train_features, train_outcome - training data (pd.DataFrame)
     test_features, test_outcome - test data (pd.DataFrame) 
    '''
    # Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    results = pd.DataFrame(columns=["Model","Parameters",
                                    "MSE","MAE"])

    # Loop over models 
    for model_key in models.keys(): 
        
        # Loop over parameters 
        for idx,params in enumerate(params_grid[model_key]): 
            
            # Create model 
            model = models[model_key]
            model.set_params(**params)

            # Start timing for fit
            startmodel = datetime.datetime.now()
            print("\tTraining:", model_key, "|", params)

            # Fit model on training set 
            model.fit(train_features, train_outcome)

            # Predict on testing set 
            test_pred = model.predict(test_features)

            # Save results
            dump(model, save_path+f"/{model_key} - Model {idx} - {validation_low} {validation_high}.joblib")
            dump(test_pred, save_path+f"/{model_key} - Predictions {idx} - {validation_low} {validation_high}.joblib")

            # Finish timing
            endmodel = datetime.datetime.now()
            print("\tTime elapsed to train and predict: ",endmodel-startmodel,"\n")

            # Evaluate predictions 
            MSE = mean_squared_error(test_outcome,test_pred)
            MAE = mean_absolute_error(test_outcome,test_pred)

            # Store results in your results data frame 
            newrow = pd.DataFrame([[model_key,params,MSE,MAE]],
                                   columns=["Model","Parameters",
                                            "MSE","MAE"])
            results = results.append(newrow)
            
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed For All Fitting and Prediction:", stop - start)

    return results        