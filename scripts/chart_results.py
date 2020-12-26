### PLOTTING FUNCTIONS ONLY SUPPORT 1 TARGET VARIABLE PER RUNs

import pandas as pd
import numpy as np
import joblib as jl
# import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from os import listdir
from os.path import join
# import pipeline
import utils
import fit_models_without_pca
import fit_models_with_pca

OUTPUT = "../output"
DATA = "../output/data"

LINE_PLOT_LABELS = {
	'observed_retail_and_recreation_percent_change_from_baseline': 'Retail and Recreation'
	}


fips_lst = ['06059', '36047', '17031', '13121']

pca_mods = ['AdaBoostRegressor - Predictions 5.joblib',
			'KNeighborsRegressor - Predictions 5.joblib',
			'Ridge - Predictions 9.joblib',
			'RandomForestRegressor - Predictions 5.joblib',
			'LinearSVR - Predictions 2.joblib']
nopca_mods = ['AdaBoostRegressor - Predictions 2.joblib',
			  'Lasso - Predictions 1.joblib',
			  'LinearRegression - Predictions 0.joblib',
			  'RandomForestRegressor - Predictions 2.joblib']


def load_df(target, df_type='Validation', df_cv='0.8 0.9', pca=False):
	"""
	loads validation or target data from output/data folder
	inputs:
		target: list of target variables
		df_type: string ("Valication" or "Test")
		pca: bool - if you want to load pca or nopca files
		df_cv:  string representing cv slice
	returns:
		df with fips, date, and target variables for df_type
	"""

	assert df_type in ('Validation', 'Test'), "df_type must be Validation or Test"
	assert isinstance(target, list), 'target must be a list of strings'

	print("loading df...")
	pca = ' PCA' if pca else ''

<<<<<<< HEAD
	feats = jl.load('../output/data/Data - {} Features{} 0.8 0.9.joblib'.format(
		df_type, pca))
	targets = jl.load('../output/data/Data - {} Target 0.8 0.9.joblib'.format(df_type))
=======
	if df_type == "Validation":

		feats = jl.load('../output/data/Data - {} Features{} {}.joblib'.format(
			df_type, pca, df_cv))
		targets = jl.load('../output/data/Data - {} Target {}.joblib'.format(df_type, df_cv))

	elif df_type == "Test":

		feats = jl.load('../output/data/Data - {} Features{}.joblib'.format(
			df_type, pca))
		targets = jl.load('../output/data/Data - {} Target.joblib'.format(df_type))
>>>>>>> 983bc9fe69ff90f405f22b204e6bd5f647a75ace

	feats.drop(['date', 'fips'], axis=1, inplace=True, errors='ignore')		

	for df in [feats, targets]:
		df.drop(['StateName', 'CountyName'], axis=1, inplace=True, errors='ignore')

	df = pd.concat([feats, targets], axis=1).reset_index().drop('index', axis=1)
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))
	df = df[['fips', 'date'] + target]
	names = {c: 'observed_' + c for c in target}
	df.rename(names, axis=1, inplace=True)

	return df


def load_predictions(folder, model_list, target):
	"""
	loads predictions for specifified models
	this function assumes files are named as follows:
	AdaBoostRegressor - Predictions 4.joblib
	KNeighborsRegressor - Predictions 5.joblib
	LinearSVR - Predictions 0.joblib
	inputs:
		model_list = [list of files in folder that correspond to model predictionss]
		target = [list of target variables]
	returns:
		df of 
	"""

	print("loading predictions")
	predictions = {}

	for f in model_list:
	    preds = pd.Series(jl.load(join(OUTPUT, folder, f)))
	    predictions[f[:f.find('.')-2]] = preds

	df = pd.DataFrame(predictions)

	for t in target:
		names = {c: '_'.join([c, t]) for c in predictions}

	df.rename(names, axis=1, inplace=True)

	return df


def create_prediction_df(target, model_list, df_type='Validation', df_cv='0.8 0.9', pca=False):
	"""
	returns df including target variables, date, fips and observed variables
	inputs:
		model_list = [list of files in folder that correspond to model predictionss]
		target = [list of target variables]
		pca: bool - if you want to load pca or nopca files
		df_type: string ("Valication" or "Test")
		df_cv:  string representing cv slice
	returns:
		df including target variables, date, fips and observed variables		
	"""

	assert df_type in ('Validation', 'Test'), "df_type must be Validation or Test"
	assert isinstance(target, list), 'target must be a list of strings'

	folder = 'models_predictions_pca' if pca else 'models_predictions_nopca'

	observed = load_df(target, df_type=df_type, df_cv=df_cv, pca=pca)
	predictions = load_predictions(folder, model_list, target)

	print("merging...")
	merged = observed.merge(predictions, how='inner', left_index=True, right_index=True)

	return merged
		

def load_training_county_mean(target, df_cv='0.8 0.9'):
	"""
	returns df with fips and county means of variables in targets
	inputs:
		target is a list of target variables
		df_cv:  string representing cv slice
	return:
		df with fips and county means of variables in targets
	"""
	training_df = jl.load(join(DATA, 'Data - Train Target {}.joblib'.format(df_cv)))

	# return training_df
	df = pd.DataFrame(training_df['fips'].unique())
	df.columns = ['fips']

	for t in target:
		means = training_df.groupby('fips')[t].mean()
		df = df.merge(means, how='left', on='fips').rename(
			{t: t+'_cmean'}, axis=1)

	return df


def calc_MAE_by_county(targets, training_means, predictions):
	"""
	inputs:
		target is a list of target variables
		training_means: df that includes by-county target variable means
		predictions: df that includes date, fips and prediction columns only
	returns:
		df of predictions and county means
	"""

	for t in targets:
	
		df = predictions.merge(training_means, how='left', on='fips').rename(
			{t: t+'_cmean'},
			axis=1)
		df[t + '_error'] = df.apply(
			lambda x: abs(x['observed_'+t] - x[t+'_cmean']), axis=1)
	
	return df


def overall_validation_null_mae(target):
	"""
	Calculates null MAE for each cross validation slice then
	averages them
	returns float 
	"""

	total_mae=0
	for cv in ['0.6 0.7', '0.7 0.8', '0.8 0.9']:
		means = load_training_county_mean(target, df_cv=cv)
		df = load_df(target, df_type='Validation', df_cv=cv, pca=True)
		df = df.merge(means, how='left', on='fips')
		mae = abs(df['observed_' +  target[0]] - df[target[0] + '_cmean']).mean()
		print(mae)

		total_mae += mae

	return  total_mae / 3




def test_null_mae(target):
	"""
	calculates null MAE on test slice
	"""

	means = load_training_county_mean(target)
	df = load_df(target, df_type='Test', pca=True)
	df = df.merge(means, how='left', on='fips')

	mae = abs(df['observed_' +  target[0]] - df[target[0] + '_cmean']).mean()

	return mae




def plot_predictions(df, null_models, fips_list, outpath, df_type='Validation', pca=False):
	"""
	creates by-county line plots of all pred
	inputs:
		df: prediction dataframe from create_prediction_df to outpath
		null_models: df from load_training_county_mean
		fips_list: list of fips to consider
		outpath: folder to put charts
		pca: bool - if you want to load pca or nopca files
		df_type: string ("Valication" or "Test")
	"""

	for fips in fips_list:
		print('running for fips', fips)
		county = df[df['fips'] == fips].drop('fips', axis=1)

		for target in [c for c in df.columns if c.startswith('observed')]:
			null = null_models.loc[null_models['fips']==fips, target[9:]+'_cmean'].item()

			#### add the training mean
			for_plot = county[['date'] + [c for c in county.columns if c.endswith(target[9:])]]
			for_plot['date'] = for_plot['date'].astype('O')
			plt.clf()

			fig = plt.figure(figsize=(15, 10))
			ax = fig.add_subplot(111)
			
			ax.hlines(null, for_plot['date'].min(), for_plot['date'].max(),
				color='grey', linestyle='dashed', linewidth=3, label='Null Model')
			ax.plot(for_plot['date'], for_plot[target], color='black', linewidth=4, label='Observed')

			for c in for_plot.columns:
				if 'Predictions' in c:
					label = c[:c.find('Predictions_') - 2]
					print(label)
					ax.plot(for_plot['date'], for_plot[c], linewidth=2, label=label)

			plt.title(
			    'Predictions and Observations for Select Models on {} Data\nFIPS: {}\nVariable: {}'.format(
			    	df_type, fips, LINE_PLOT_LABELS[target]),
			    fontsize=20)
			plt.xlabel('Date', fontsize=12)
			plt.ylabel('Change in Mobility From from Baseline', fontsize=12)
			ax.legend(loc='best')
			ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

			pca_title = 'pca' if pca else 'nopca'
			name = '_'.join([target, fips, df_type, pca_title])

			plt.savefig(join(outpath, '_'.join([target, 'linechart', fips, pca_title, df_type, 'globalMAE'])))


			plt.close()



def plot_test_barchart():
	"""
	plots bar charts for test data
	takes no parameters and isn't flexible but god damn it gets the job done
	"""

	results = pd.read_csv((join(OUTPUT, 'test_MAEs.csv')))
	results['best'] = results.groupby('Model')['MAE'].transform(min)
	results = results.loc[results['best'] == results['MAE'], ['Model', 'MAE']].drop_duplicates(ignore_index=True)
	null = calc_global_null(['retail_and_recreation_percent_change_from_baseline'])
	results = results.append({'Model': 'Global Mean', 'MAE': null}, ignore_index=True)

	print('updates working')
	# sns.set(rc={'figure.figsize': (15, 8)})
	sns.set_style("white")
	f, ax = plt.subplots(figsize=(18, 8))
	ax = sns.barplot(x='Model', y='MAE', data=results)
	plt.xlabel("Best Models", fontsize=13)
	plt.title('Mean Absolute Error on Test Data for Best Models by Type', fontsize=18)
	# plt.xticks(rotation=45)
	# plt.legend(fontsize=20)
	plt.savefig('../output/plot/results/mae_test_bar.png')
	return results

	# return results





def plot_model_results(results, targets, null_MAE, outpath, val_or_test='Validation', pca=False):
	"""
	creates swarmplots in output/plots
	inputs:
		results: dataframe of results by model
		target: [list of target variables]
		null_MAE: float
		outpath: folder to put charts
		val_or_test: string that is either "Validation" or "Test"
		pca: bool - if you want to load pca or nopca files
	"""


	for t in targets:
		plt.clf()

		f = plt.figure(figsize=(15, 5))
		sns.swarmplot(x='Model', y='MAE', data=results)
		plt.title('Mean Absolute Error by Model on {} Data'.format(val_or_test), fontsize=20)
		plt.xlabel('Model', fontsize=14)
		plt.ylabel('Mean Absolute Error', fontsize=14)
		# print(county_means.columns)
		plt.hlines(y=null_MAE, xmin=-1, xmax=results.Model.nunique(),
			linestyle='dashed', linewidth=3, label='Global Mean', color='grey')
		plt.text(4.2, null_MAE+.4, 'Global Mean',
			fontsize=12, color='grey')

		pca_name = 'pca' if pca else 'nopca'

		plt.show()
		plt.savefig(join(outpath, t + '_MAE_by_model_' + pca_name + val_or_test + 'globalMAE' + '.png'))
		plt.close()


def execute_plots(targets, fips_list):
	"""
	runs code to generate plots of results
	inputs:
		targets (list): variables names corresponding to target
		fips_list (list): list of pips codes to make counties for
	saves plots to output/plots
	"""

	model_list = {True: pca_mods, False: nopca_mods}

	null_model = load_training_county_mean(targets)
	outpath = join(OUTPUT, 'plot', 'results')

	# training_means = load_training_county_mean(targets)

	# for p in (True):
	p = True

	print('model list', model_list[p])

<<<<<<< HEAD
		if p:
			results = pd.read_csv(join(OUTPUT, 'model_validation_results_with_pca_average.csv'))
		else:
			results = pd.read_csv(join(OUTPUT, 'model_validation_results_without_pca_average.csv'))
=======
	pred_df = create_prediction_df(targets, model_list[p], df_type='Validation', pca=p)
	plot_predictions(pred_df, null_model, fips_list, outpath, df_type='Validation', pca=p)
>>>>>>> 983bc9fe69ff90f405f22b204e6bd5f647a75ace

	# null_MAE = overall_validation_null_mae(targets)
	null_MAE = calc_global_null(targets)
	if p:
		results = pd.read_csv(join(OUTPUT, 'model_validation_results_with_pca_average.csv'))
	else:
		results = pd.read_csv(join(OUTPUT, 'model_validation_results_without_pca_average.csv'))

	plot_model_results(results, targets, null_MAE, outpath, pca=p)


def execute_test_plots(targets, fips_list):
	"""
	runs code to generate plots of results for test data
	inputs:
		targets (list): variables names corresponding to target
		fips_list (list): list of pips codes to make counties for
	saves plots to output/plots
	"""


	model_list = pca_mods
	null_model = load_training_county_mean(targets)
	outpath = join(OUTPUT, 'plot', 'results')

	print('model list', model_list)
	pred_df = create_prediction_df(targets, model_list, df_type='Test', pca=True)
	plot_predictions(pred_df, null_model, fips_list, outpath, df_type='Test', pca=True)

	null_MAE = calc_global_null(targets)

	results = pd.read_csv((join(OUTPUT, 'test_MAEs.csv')))
	plot_model_results(results, targets, null_MAE, outpath, val_or_test='Test', pca=True)


<<<<<<< HEAD
def mae_bar():
	fig, ax = plt.subplots()
	with_pca = pd.read_csv('../output/model_validation_results_with_pca_average.csv')
	without_pca = pd.read_csv('../output/model_validation_results_without_pca_average.csv')
	with_pca['PCA'] = 'PCA'
	without_pca['PCA'] = 'No PCA'
	with_pca_min = with_pca.groupby('Model').min()
	without_pca_min = without_pca.groupby('Model').min()
	mae = pd.concat([with_pca_min, without_pca_min])
	sns.set(rc={'figure.figsize':(18, 5)})
	ax = sns.barplot(x=mae.index, y='MAE', hue="PCA", data=mae)
	plt.xlabel("Best Models", fontsize=13)
	plt.title('Mean Absolute Error for Best Models by Type', fontsize=18)
	plt.legend(fontsize=20)
	plt.savefig('../output/plot/results/mae_bar.png')
=======
def calc_global_null(target):
	"""
	calculates global mean for training data
	"""
	training_df = jl.load(join(DATA, 'Data - Train Target 0.8 0.9.joblib'))
	prediction = training_df[target].mean().item()
	df = load_df(target, df_type='Validation', df_cv='0.8 0.9', pca=True)
	df['error'] = abs(df['observed_'+target[0]] - prediction)
		
	return df['error'].mean()
>>>>>>> 983bc9fe69ff90f405f22b204e6bd5f647a75ace




	