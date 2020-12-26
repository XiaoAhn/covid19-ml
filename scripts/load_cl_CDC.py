import pandas as pd
import numpy as np
from os.path import join, abspath, dirname, exists
import utils

ROOT = join(dirname(dirname(abspath(__file__))))
RAW = join(ROOT, 'data_raw')
INT = join(ROOT, 'data_intermediate')
# FEAT = join(ROOT, 'clean_features')



def read_CDC():
	"""
	reads and returns dfs for cdc cases and deaths
	"""

	cases = pd.read_csv(join(RAW, 'cdc_covid_confirmed_usafacts.csv'))
	deaths = pd.read_csv(join(RAW, 'cdc_covid_deaths_usafacts.csv'))

	return cases, deaths


def cl(df, val_name):
	"""
	transforms df to long format with rows identified by date-county
	df: dataframe with counties as rows and dates as columns
	val_name: string of variable name being transposed
	"""

	df = df[df['countyFIPS'] != 0]
	df.drop(['County Name', 'State', 'stateFIPS'], axis=1, inplace=True)
	df = df.melt(id_vars='countyFIPS', var_name='date', value_name=val_name)

	return df


def main():
	"""
	executes steps to create intermediate cd deaths and cases csvs
	"""

	cases, deaths = read_CDC()
	cases = cl(cases, 'cases')
	deaths = cl(deaths, 'deaths')

	### create features
	for df, var in zip((cases, deaths), ('cases', 'deaths')):

		for window in [3, 7]:
			df['{}_{}d_avg'.format(var, window)] = df.groupby('countyFIPS')[var].transform(
				lambda x: x.rolling(window, 1).mean())

	# cases['log_cases'] = np.log(cases['cases'])
	# deaths['log_deaths'] = np.log(deaths['deaths'])

	cases.to_csv(join(INT, 'CDC_cases.csv'), index=False)
	deaths.to_csv(join(INT, 'CDC_deaths.csv'), index=False)

	return cases, deaths