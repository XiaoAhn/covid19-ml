# covid19-ml-project

### Directory Description
- scripts: contains '.py' files that run the model
- data_raw:  contains files directly pulled from online sources
- data_intermediate:  contains cleaned files ready to be integrated into main data frame
- output:  contains folders that describe results from the gridsearch
  - data: dfs used for most recent gridsearches (train, validation, test)
  - plot: plots describing the output of the models
  - models_predictions_nopca:  predictions and model objects for various hyper-parameters
  - models_predictions_pca:  predictions and model objects for various hyper-parameters
  - *Note the previous two folders are empty because much of their contents was far too large to upload, see instructions for how to recreate*
  
  
### Description of Files

#### scripts
- building_graphs.py:  code to create various data exploration graphs
- build_master_df.py:  assembles df to be used in model
- chart_results.py:  contains functions that build select charts from results
- create_intermediate_data.py:  contains function to populate intermediate data folder
- fit_models_with_pca.py:  functions to prepare and execute gridsearch with pca
- fit_models_without_pca.py:  functions to prepare and execute gridsearch without pca
- identify_key_features.py:  identifies key features in non-pca model
- import_health.py:  creates intermediate data csv for cdc health characteristics
- load_cl_CDC.py:  creates intermediate data csv for cases and deaths
- load_cl_target.py:  creates intermediate data csv for target variables
- load_interventions.py:  creates intermediate data csv for county-level covid interventions
- pipeline.py:  various functions for preparing for grid search
- pull_census.py:  creates intermediate data csv for ACS and NAICS data from census sources
- pull_noaa.py:  creates intermediate data csv for noaa weather data
- pull_votes.py:  creates intermediate data csv for MIT Election Data
- utils.py:  various utility functions

#### Instructions
**WARNING:  OUR CODE TAKES A LONG TIME TO RUN**
**PROCEED WITH CAUTION**

0. read our awesome report
1. run setup.sh from the main project folder to unzip data files and setup your virtual environment
2. to recreate files in intermediate data, run create_intermediate_data.populate_intermediate_data() - otherwise simply use the files provided in the zipped folder
3. to run our grid search, execute either fit_models_with_pca.py or fit_models_without_pca.py, these files will populate  certain sections of the output folder with model objects, predictions, and scores on validation data - **takes a long time**
4. to score models on our test data, run evaluate_models_on_test.py - this will create files corresponding to the predictions for the test period of each model in the models_predictions_nopca and models_predictions_pca folders
5. play around with functions in chart_results.py to... chart the results.  execute_plots and execute_test plots can be called to track model performance for different fips codes and target variables.  other functions allow experimenting with different null models as benchmarks

