import build_master_df
import pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_master_df():
    """
    Function to build master dataset by calling 'build_master_df' py file
    """
    raw = build_master_df.build_df()
    return raw

def split_master_df(raw, save_path, validation_low, validation_high):
    """
    Function to get train / validation / test sets.
    """

    # Identify variables
    index_vars   = ['StateName','CountyName','fips','date']
    target_vars  = [col for col in raw.columns if (col.endswith('change_from_baseline'))]
    main_target  = 'retail_and_recreation_percent_change_from_baseline'
    features     = [col for col in raw.columns if (col not in index_vars) and (col not in target_vars)]

    # Get full dataset for use and drop unnecessary variables
    df = raw.dropna(subset=[main_target])

    # Split train test 
    train_full,validation_full,test_full = pipeline.get_train_test(df, test_size=0.1,
                                                                    time_series=True, validation=True,
                                                                    validation_low=validation_low,
                                                                    validation_high=validation_high)
    train_target = train_full[main_target]
    validation_target = validation_full[main_target]
    test_target = test_full[main_target]
    train_features = train_full[features]
    validation_features = validation_full[features]
    test_features = test_full[features]

    # Impute and normalize
    train_features,validation_features,test_features = pipeline.impute_missing(train_features,test_features,
                                                                                validation_features,how='median')
    train_features,validation_features,test_features = pipeline.normalize_vars(train_features,test_features,
                                                                                validation_features)

    # Make dfs to save in output
    train_out = pd.concat((train_full[index_vars], train_features), axis=1)
    validation_out = pd.concat((validation_full[index_vars], validation_features), axis=1)
    test_out = pd.concat((test_full[index_vars], test_features), axis=1)
    train_out_target = pd.concat((train_full[index_vars], train_target), axis=1)
    validation_out_target = pd.concat((validation_full[index_vars], validation_target), axis=1)
    test_out_target = pd.concat((test_full[index_vars], test_target), axis=1)

    # Save output
    dump(train_out, save_path+f"/Data - Train Features {validation_low} {validation_high}.joblib")
    dump(validation_out, save_path+f"/Data - Validation Features {validation_low} {validation_high}.joblib")
    dump(test_out, save_path+f"/Data - Test Features.joblib")
    dump(train_out_target, save_path+f"/Data - Train Target {validation_low} {validation_high}.joblib")
    dump(validation_out_target, save_path+f"/Data - Validation Target {validation_low} {validation_high}.joblib")
    dump(test_out_target, save_path+f"/Data - Test Target.joblib")                              

    return train_features,validation_features,test_features,train_target,validation_target,test_target

def fit_and_eval_models(train_features,train_target,
                        validation_features,validation_target,
                        save_path,validation_low,validation_high):
    """
    Function to fit a number of regression models,
    evaluate, and return the results. This takes a while to run.
    """

    # Config: Dictionaries of models and hyperparameters
    MODELS = {
        'LinearRegression': LinearRegression(), 
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor(),
        'AdaBoostRegressor':AdaBoostRegressor(),
    }
    GRID = {
        'LinearRegression': [{}],
        'Lasso': [{'alpha':x, 'random_state':0, 'max_iter':10000} for x in [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]],
        'RandomForestRegressor': [{'n_estimators':x, 'max_features':'log2',
                                   'n_jobs':-1} \
                                    for x in [100,500,1000]],
        'AdaBoostRegressor': [{'n_estimators':y} for y in [50,75,100,125,150,175,200]],
    }

    # Fit and get results
    model_results = pipeline.build_regressors(MODELS, GRID,
                                              train_features, train_target,
                                              validation_features, validation_target,
                                              save_path, validation_low, validation_high)
    return model_results


if __name__ == "__main__":

    # Do cross-validation
    full_results = pd.DataFrame()
    raw = get_master_df()
    for val_cutoffs in [(0.6,0.7),(0.7,0.8),(0.8,0.9)]:
        train_features,validation_features,test_features,train_target,validation_target,test_target = split_master_df(raw, "../output/data",
                                                                                                                        val_cutoffs[0], 
                                                                                                                        val_cutoffs[1])
        model_results = fit_and_eval_models(train_features,train_target,
                                            validation_features,validation_target,
                                            "../output/models_predictions_nopca",
                                            val_cutoffs[0], val_cutoffs[1])
        full_results = pd.concat((full_results,model_results), axis=0)

    # Sort and save results
    full_results = full_results.sort_values('MAE')
    full_results.to_csv("../output/model_validation_results_without_pca_all_validations.csv")

    # Get average results
    full_results['Parameters'] = full_results['Parameters'].astype(str)
    full_results = full_results.groupby(by=['Model','Parameters']).mean().reset_index()
    full_results = full_results.sort_values('MAE')
    full_results.to_csv("../output/model_validation_results_without_pca_average.csv")
