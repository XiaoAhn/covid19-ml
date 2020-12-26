"""
pulls census data
it takes a long time to run full file - best to run these functions seperately
in a jupyter notebook
"""

import requests as re
import pandas as pd
import numpy as np
from os.path import join, abspath, dirname

NAICS_ENDPOINT = r"https://api.census.gov/data/2017/cbp?get=NAICS2017,NAICS2017_LABEL,GEO_ID,EMP&for=county:*"
ACS_ENDPOINT = r"https://api.census.gov/data/2018/acs/acs5/subject?get=NAME,VARS&for=county:*"
ROOT = join(dirname(dirname(abspath(__file__))))
OUTPATH = join(ROOT, 'data_intermediate')
RAW = join(ROOT, 'data_raw')



def pull_data(endpoint):
	"""
	requests from API and returns pd.DataFrame
	"""
	print('pulling', endpoint)
	r = re.get(endpoint)
	return pd.DataFrame(r.json())


def write(df, fn):
	"""
	writes df to OUTPATH folder
	"""

	df.to_csv(join(OUTPATH, fn), index=False)
	


### ACS functions
def read_acs_vars(fn):
	"""
	reads txt files of vars and contructs endpoint
	vars must be stored in seperate files because there is a limit
	to the number of variables one can request from the census with
	one pull
	inputs: filename
	returns (str): endpoint
	"""
	path = join(ROOT, 'data_raw', fn)
	with open(path, 'r') as f:
		lines = f.readlines()

	s = ','.join(list(map(lambda x: x.strip(), lines)))
	return ACS_ENDPOINT.replace('VARS', s)


def cl_ACS_col(col):
	"""
	converts all values in col to numeric and removes negatives
	(because negatives signal no data)
	inputs:
		col: pd.Series
	returns cleaned pd.Series
	"""
	new_col = col.copy()
	new_col = pd.to_numeric(new_col)
	new_col[new_col < 0] = np.nan
	return new_col


def clean_ACS(df):
	"""
	executes cleaning steps for df of ACS vars
	returns cleaned dataframe
	"""

	df.columns = df.iloc[0]
	df = df[1:]
	data_cols = [c for c in df.columns if c not in ['NAME', 'county', 'state', 'STATE', 'COUNTY']]
	df[data_cols] = df[data_cols].apply(cl_ACS_col, axis=0)

	### remove empty cols
	wanted_cols = df.isnull().sum()!=df.shape[0]
	df = df[wanted_cols.index[wanted_cols]]

	return df


def acs_feature_creating(df):
	"""
	a nightmare of a function that generates features from census
	data and returns the df
	"""

	df = df.copy()

	df['fips'] = df['state'] + df['county']

	df['log_pop'] = np.log(df['S0101_C01_001E'])
	df['log_households'] = np.log(df['S1101_C01_001E'])

	### commuting to work vars
	df['workers_pct'] = df['S0801_C01_001E'] / df['S0101_C01_001E']
	commuter_cols = ['S0801_C01_002E', 'S0801_C01_009E', 'S0801_C01_010E',
					 'S0801_C01_011E', 'S0801_C01_012E', 'S0801_C01_013E']
	for c in commuter_cols:
		df[c] = df[c] / df['S0801_C01_001E'] ## dividng by number of commuters

	### dividing housing type by total households
	df['households_u18_pct'] = df['S1101_C01_010E'] / df['S1101_C01_001E']
	df['households_o60_pct'] = df['S1101_C01_011E'] / df['S1101_C01_001E']
	df['household_owner_occupied_pct'] = df['S1101_C01_019E'] / df['S1101_C01_001E']
	df['household_rented_pct'] = df['S1101_C01_020E'] / df['S1101_C01_001E']

	### diving income brackets by total households
	income_cols = (['S1901_C01_002E', 'S1901_C01_003E', 'S1901_C01_004E', 'S1901_C01_005E', 
	'S1901_C01_006E', 'S1901_C01_007E', 'S1901_C01_008E',
	'S1901_C01_009E', 'S1901_C01_010E', 'S1901_C01_011E'])
	for c in income_cols:
		df[c] = df[c] / df['S1101_C01_001E']

	### food stamps
	df['food_stamps_pct'] = df['S2201_C03_001E'] / df['S1101_C01_001E']

	df.drop(['S0801_C01_001E', 'S1101_C01_010E',
		'S1101_C01_011E', 'S1101_C01_019E', 
		'S1101_C01_020E', 'S2201_C03_001E'],
		axis=1, inplace=True, errors='raise')

	return df


def rename_acs_cols(df):
	"""
	renames select columsn in acs dataframe based on 'rename_ACS_vars.txt'
	"""

	### grab acs columns
	with open(join(RAW, 'rename_ACS_vars.txt'), 'r') as f:
		d = f.readlines()

	renames = {}
	for r in d:
		items = r.split(':')
		k = items[0].strip()
		v = items[1].strip()
		renames[k] = v

	df.rename(renames, axis=1, inplace=True, errors='raise')
	return df


def interpolate_with_state(df):
	"""
	interpolates missing values with state means
	"""
	df = df.copy(deep=True)
	for c in df.columns:

		### first fill in with state
		if df[c].isnull().sum() > 0:
			df['interpolate_val'] = df.groupby('state')[c].transform(np.mean)
			df[c].fillna(df['interpolate_val'], inplace=True)

		### fill in remainder with 0
		df[c].fillna(0, inplace=True)

	df.drop('interpolate_val', axis=1, inplace=True, errors='raise')
	return df


def execute_ACS():
	"""
	Executes steps required to pull ACS
	"""

	final_df = pd.DataFrame()
	for fn in ['acs_vars0.txt', 'acs_vars1.txt']:
		endpoint = read_acs_vars(fn)
		df = pull_data(endpoint)
		df = clean_ACS(df)
		if final_df.empty:
			final_df = df
		else:
			final_df = final_df.merge(df, how='outer', on=['state', 'county', 'NAME'])

	final_df = acs_feature_creating(final_df)
	final_df = rename_acs_cols(final_df)
	final_df = interpolate_with_state(final_df)
	write(final_df, 'ACS.csv')

	return final_df


### NAICS FUNCTIONS
def clean_NAICS(df):
	"""
	executes cleaning steps for business pattern df
	"""

	df.columns = df.iloc[0]
	df = df[1:]
	df['EMP'] = df['EMP'].astype(int)
	df['fips'] = df['GEO_ID'].str[-5:]
	df.rename({'NAICS2017': 'Industry', 
	               'NAICS2017_LABEL': 'Industry_Label'}, axis=1, inplace=True)

	return df


def merge_on_totals(df, level):
	"""
	merges total number of workers by industry
	inputs:
		df: dataframe of NAICS data generated from pull_data and clean NAICS
		level:  digits of NAICS code at which we want to aggregate
	"""

	totals = df.loc[df['Industry']=='00', ['fips', 'EMP']]
	secs = df[df['Industry'].str.len()==level]
	merged = secs.merge(totals, how='left', left_on='fips', right_on='fips')
	merged.rename({'EMP_x': "employee_num",
               'EMP_y': 'employee_num_county'}, axis=1, inplace=True)
	merged = merged[merged['Industry'] != '00']
	merged = merged[(['Industry', 'Industry_Label', 'employee_num', 
				   'employee_num_county', 'fips'])]
	merged['emp_pct'] = merged['employee_num'] / merged['employee_num_county']
	return merged


def reshape(df):
	"""
	reshapes dataframe such that there are columns for each industry
	"""
	return df.pivot(index='fips', columns='Industry_Label',
			values='emp_pct').reset_index()


def rename_naics(df):
	"""
	renames columns
	"""
	new_names = {c: "NAICS_" + c for c in df.columns if c not in ('fips', 'state')}
	df.rename(new_names, axis=1, inplace=True, errors='raise')
	return df
	


def execute_NAICS():
	"""
	executes steps to make NAICS intermediate data 
	"""

	df = pull_data(NAICS_ENDPOINT)
	df = clean_NAICS(df)
	df = merge_on_totals(df, 2)
	df = reshape(df)
	df['state'] = df['fips'].str[:2]
	df = interpolate_with_state(df)
	df = rename_naics(df)
	write(df, 'NAICS.csv')

	return df










