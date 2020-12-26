import pandas as pd
import numpy as np
import joblib as jl
from os import listdir
from os.path import join

OUTPUT = "../output"
DATA = "../output/data"



def get_models_from_folder(pca=True):
	"""
	returns list of directories to model files
	"""

	folder = 'models_predictions_pca' if pca else 'models_predictions_nopca'
	models = ([join(OUTPUT, folder, m) for m in listdir(join(OUTPUT, folder))
		if 'Model' in m and '0.9' in m])

	return models


def get_test_predictions_files(pca=True):
	"""
	returns list of files that have information on predictions
	for test data	
	"""

	folder = 'models_predictions_pca' if pca else 'models_predictions_nopca'
	predictions = [join(OUTPUT, folder, p) for p in listdir(join(OUTPUT, folder)) \
		if 'Predictions' in p and 'Test' in p]

	return predictions


def load_test_data(pca=True):
	"""
	returns df of test features - for pca colnames will be unnamed
	"""

	flist = [f for f in listdir(DATA) if 'Test' in f]
	if pca:
		flist = [f for f in flist if 'PCA' in f]


	if len(flist) != 1:
		raise Exception("File is not uniquely identified")

	feat_file = flist[0]
	test_feats = jl.load(join(DATA, feat_file))
	return test_feats


def load_test_target():
	"""
	returns test target
	"""
	return jl.load(join(DATA, 'Data - Test Target.joblib'))



def generate_prediction(model, test_feats):
	"""
	returns array of predictions for model on test data
	inputs
		model: string representing a model joblib file
		test_feats: df of test features
	"""
	print('predicting for', model)
	model = jl.load(model)
	predictions = model.predict(test_feats)
	return predictions


def generate_all_predictions(model_list, test_feats):
	"""
	calculates predictions for all models in folder
	inputs:
		model_list: list of model paths
		test_feats: df of test features
	returns

	"""

	df = pd.DataFrame

	for m in model_list:

		predictions = generate_prediction(m, test_feats)
		path = get_save_path(m)
		jl.dump(predictions, path)


def execute_all_predictions():
	"""
	calculates predictions for pca and non pca data
	!!!! not working for non-pca - something about the file path name
	not enough time to fix
	"""

	for pca in (True, False):
		test_data = load_test_data(pca)
		model_list = get_models_from_folder(pca)

		generate_all_predictions(model_list, test_data)

	print('done')


def get_save_path(m):
	"""
	returns name for output file as a function of model filename
	"""

	p = m.replace('Model', 'Predictions')
	p = p[:p.find('0.8')] + 'Test.joblib'

	return p


def calc_MAE(test_target, predictions, var):
	"""
	returns mean absolute error for predictions on test_target
	var allows for flexibility in target variable
	test_target: df with observed values for target var
	"""

	mae = abs(test_target[var] - predictions).mean()

	return mae


def calc_MAE_by_model(prediction_list, test_target, pca=True):
	"""
	returns df with columns for model and MAE
	inputs:
		prediction_list: list of prediction filenames
		test_target: df with observed values for target var
	"""

	maes = []
	var = 'retail_and_recreation_percent_change_from_baseline'

	name_cutoff = 33 if pca else 35

	for p in prediction_list:
		prediction = jl.load(p)
		n = p[name_cutoff:p.find(' - Test')]

		mae = calc_MAE(test_target, prediction, var)
		maes.append((n, mae))

	df = pd.DataFrame.from_records(maes)
	df.columns = ['Model', 'MAE']

	return df


def execute_MAE_cal():
	"""
	executes process of create MAE for all predictions
	returns df of MAEs per model and saves csv in output folder
	"""

	test_target = load_test_target()
	prediction_list = get_test_predictions_files(pca=True)
	rv = calc_MAE_by_model(prediction_list, test_target)
	rv['version'] = rv['Model'].str[-1]
	rv['Model'] = rv['Model'].apply(lambda x: x[:x.find(' - ')])

	print('outputting csv...')
	rv.to_csv(join(OUTPUT, 'test_MAEs.csv'), index=False)

	return rv


if __name__ == '__main__':

	print('whaddup')
	execute_MAE_cal()




