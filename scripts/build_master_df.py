import numpy as np
import pandas as pd
import geopandas as gpd
import read_file
import import_health
import datetime
import warnings
warnings.filterwarnings('ignore')

def build_df():
	"""
	Assembles all components into 1 large dataframe
	"""

	### Target variable
	print('Reading Google mobility data...')
	df = read_file.read_target()

	### Shape file for area
	print('Reading shape file (takes a couple minutes)')
	geodf = gpd.read_file('../data_raw/tl_2017_us_county.shp')
	geodf['area'] = geodf.geometry.apply(lambda x: x.area)
	geodf['fips'] = geodf['STATEFP'] + geodf['COUNTYFP']
	print('\tMerging on areas')
	df = df.merge(geodf[['fips', 'area']], how='left', on='fips')

	### NAICS data
	print('Reading / merging NAICS business pattern data...')
	NAICS = read_file.read_NAICS()
	df = df.merge(NAICS, how='left', on='fips')
	
	### Census data
	print('Reading / merging ACS census data...')
	ACS = read_file.read_ACS()
	df = df.merge(ACS, how='left', on='fips')

	### CDC general health data
	print('Reading / merging CDC health data...')
	health_fips,health_st = read_file.read_health()
	df = df.merge(health_fips, how='left', on='fips')
	df = df.merge(health_st, how='left', on='StateFIPS')
	print('    Interpolating missing CDC county data with state data...')
	for c in df.columns:
		if c.startswith('Percent') and not c.endswith('state'):
			df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), c + '_state']
	print('    Dropping extra CDC columns...')
	cols = [c for c in df.columns if c.endswith('_state') or c[:4] in ('CBSA', 'FIPS')]
	df.drop(columns=cols + ['NAME', 'county'], axis=1, inplace=True)

	### CDC case / death data
	print('Reading / merging CDC cases and death data...')
	cases,deaths = read_file.read_CDC()
	df = df.merge(cases, how='left', on=['date', 'fips'])
	df = df.merge(deaths, how='left', on=['date', 'fips'])

	### NOAA
	print('Reading / merging NOAA weather data...')
	noaa = read_file.read_noaa()
	df = df.merge(noaa, how='left', on=['fips', 'date'])
	print('\tInterpolating missing weather data')
	for c in df.columns:
		if c.startswith('TM') or c == 'PRCP':
			print("\tInterpolating {}...".format(c))
			vals = df.groupby(['StateFIPS', 'date'])[c].transform(np.mean)
			df[c].fillna(vals, inplace=True)
	print('\tCreating precipiation dummy...')
	df['precip_dummy'] = 0
	df.loc[df['PRCP'] > .05, 'precip_dummy'] = 1 

	### Interventions
	print('Reading interventions data...')
	interventions = read_file.read_interventions()
	df = df.merge(interventions, on='fips', how='left')
	df.drop(['STATE', 'AREA_NAME', 'StateFIPS','log_cases','log_deaths'], axis=1, inplace=True, errors='raise')
	print('\tTransforming intervention columns...')
	for c in df.columns:
		if c.startswith("int_date_"):
			print('\tTransforming {}...'.format(c))
			df[c].fillna(800000, inplace=True) ### arbitrary high date
			df[c] = df[c].apply(lambda x: datetime.date.fromordinal(int(x)))
			df[c] = df.apply(lambda x: x[c] <= x['date'], axis=1).astype('int')

	### Vote share
	print('Reading vote share data...')
	votes = read_file.read_votes()
	df = df.merge(votes, how='left', on='fips')

	## Making additional features
	df = make_features(df)

	# Drop excess columns
	df = drop_features(df)

	print('Outputting csv..')
	df.to_csv('../output/data/full_df.csv', index=False)

	return df


def make_features(df):
	df['pop_density'] = df['pop'] / df['area']
	df['cases_per_pop'] = df['cases'] / df['pop']
	df['cases_per_area'] = df['cases'] / df['area']
	df['deaths_per_pop'] = df['deaths'] / df['pop']
	df['deaths_per_area'] = df['deaths'] / df['area']

	### Weekday
	df['dayofweek'] = df['date'].apply(lambda x: x.dayofweek)
	week_dummies = pd.get_dummies(df['dayofweek'], prefix='dayofweek')
	for c in week_dummies.columns:
		df[c] = week_dummies[c]

	return df


def drop_features(df):

	df.drop([c for c in df.columns if c.startswith('chg')],
		axis=1, inplace=True, errors='raise')
	df.drop(columns=['state_x','state_y','CountyFIPS',
					 'totalvotes','dayofweek'], inplace=True)
	df.drop(columns=[col for col in df.columns if col.startswith('lag1')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.startswith('lag3')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.startswith('lag5')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.endswith('3d_avg')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.endswith('workplaces_7d_avg')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.endswith('residential_7d_avg')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.endswith('parks_7d_avg')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.endswith('grocery_7d_avg')], inplace=True)
	df.drop(columns=[col for col in df.columns if col.endswith('transit_7d_avg')], inplace=True)

	return df