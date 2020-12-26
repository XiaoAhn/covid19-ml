"""
Just some functions that we can use for easiliy loading data into our
jupyter notebooks without referencing all these folders
"""
import pandas as pd
from os.path import join, dirname, abspath
import utils

ROOT = join(dirname(dirname(abspath(__file__))))
INT = join(ROOT, 'data_intermediate')
RAW = join(ROOT, 'data_raw')

def read_target():
	"""
	loads google mobility data merged with FIPS
	"""
	df = pd.read_csv(join(INT, 'us_mobility.csv'))
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))
	df['StateFIPS'] = df['StateFIPS'].apply(lambda x: utils.prepend_0s(str(x), 2))
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

	return df


def read_ACS():
	"""
	load ACS data from intermediate folder
	"""

	df = pd.read_csv(join(INT, "ACS.csv"))
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))

	return df


def read_NAICS():
	"""
	loads NAICS file from intermediate folder
	"""
	df = pd.read_csv(join(INT, "NAICS.csv"))
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))
	df.drop(columns=['state'], inplace=True)
	for col in set(df.columns):
		if col not in ['fips']:
			df.rename(columns={col:f"NAICS {col}"}, inplace=True)
	return df


def read_noaa():
	"""
	load NOAA data from intermediate folder
	"""
	df = pd.read_csv(join(INT, 'noaa.csv'))
	df['fips'] = df['fips'].apply(
		lambda x: utils.prepend_0s(str(x), 5))	
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

	return df

def read_CDC():
	"""
	load CDC deaths and cases intermediate data folder
	"""
	cases = pd.read_csv(join(INT, "CDC_cases.csv"))
	deaths = pd.read_csv(join(INT, "CDC_deaths.csv"))

	for df in (cases, deaths):
		df['fips'] = df['countyFIPS'].apply(
			lambda x: utils.prepend_0s(str(x), 5))
		df.drop('countyFIPS', axis=1, inplace=True)
		df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
	
	return cases, deaths


def read_votes():
	"""
	load election data from intermediate data folder
	"""

	df = pd.read_csv(join(INT, 'votes.csv'))
	df['fips'] = df['fips'].apply(
		lambda x: utils.prepend_0s(str(x)[:str(x).find('.')], 5))

	return df


def read_health():
	"""
	loads health data from intermediate folder
	"""

	health_fips = pd.read_csv(join(INT, 'health_fips.csv'))
	health_fips['fips'] = health_fips['FIPS_ID'].apply(lambda x: utils.prepend_0s(str(x), 5))

	health_fips.loc[health_fips['fips'] == '15901', 'fips'] = '15009'
	health_fips.loc[health_fips['fips'] == '51918', 'fips'] = '51570'
	health_fips.loc[health_fips['fips'] == '51918', 'fips'] = '51730'
	health_fips.loc[health_fips['fips'] == '51931', 'fips'] = '51830'
	health_fips.loc[health_fips['fips'] == '51941', 'fips'] = '51670'
	health_fips.loc[health_fips['fips'] == '51949', 'fips'] = '51620'
	health_fips.loc[health_fips['fips'] == '51953', 'fips'] = '51520'
	health_fips.loc[health_fips['fips'] == '51958', 'fips'] = '51735'

	health_st = pd.read_csv(join(INT, 'healthdf_st.csv'))
	health_st['StateFIPS'] = health_st['StateFIPS'].apply(
		lambda x: utils.prepend_0s(str(x), 2))

	cols = [c for c in health_st.columns if c.startswith('Percent') or c == 'StateFIPS']
	health_st = health_st[cols]

	name_change = {c: c + '_state' for c in health_st.columns if c.startswith('Percent')}
	health_st.rename(name_change, axis=1, inplace=True, errors='raise')

	return health_fips, health_st


def read_interventions():

	# file = 'https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/data/interventions.csv'
	# df = pd.read_csv(file)

	df = pd.read_csv(join(INT, 'interventions.csv'))

	df.rename({'FIPS': 'fips'}, axis=1, inplace=True)
	df['fips'] = df['fips'].apply(lambda x: utils.prepend_0s(str(x), 5))

	return df


