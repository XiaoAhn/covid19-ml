import pandas as pd
import numpy as np
import utils
from os.path import join, abspath, dirname, exists

ROOT = join(dirname(dirname(abspath(__file__))))
RAW = join(ROOT, 'data_raw')
INT = join(ROOT, 'data_intermediate')


def execute():

	fips = get_fips_crosswalk()
	mobility = load_mobility()
	mobility_fips = merge_mobility_fips(mobility, fips)
	moving, varnames = create_moving_avgs(mobility_fips)
	lagged = create_lag_features(moving, varnames)
	final = interpolate_rolling(lagged)

	final.to_csv(join(INT, 'us_mobility.csv'), index=False)
	return final


def get_fips_crosswalk():

	# Get FIPS / name map (county level)
	fips_cnty = pd.read_csv(join(RAW, "Crosswalk - FIPS-CountyName.csv"), dtype=str)

	# Fix fips with wrong FIPS digit numbers
	fips_cnty['StateFIPS'] = fips_cnty['StateFIPS'].apply(
		lambda x: utils.prepend_0s(str(x), 2))
	fips_cnty['CountyFIPS'] = fips_cnty['CountyFIPS'].apply(
		lambda x: utils.prepend_0s(str(x), 3))

	# Get FIPS / name map (state level)
	fips_state = pd.read_csv(join(RAW, "Crosswalk - FIPS-StateName.csv"), dtype=str)

	# Fix FIPS with wrong FIPS digit numbers
	fips_state['StateFIPS'] = fips_state['StateFIPS'].apply(
		lambda x: utils.prepend_0s(str(x), 2))

	# Merge together
	fips = fips_cnty.merge(fips_state, on='StateFIPS', how='inner')
	fips['fips'] = fips['StateFIPS']+fips['CountyFIPS']

	# Change name to uppercase
	fips['StateName'] = fips['StateName'].apply(lambda x: x.upper())
	fips['CountyName'] = fips['CountyName'].apply(lambda x: x.upper())

	return fips


def load_mobility():

	# Import and subset data
	mobility = pd.read_csv(join(RAW, "google_mobility.csv"), low_memory=False)
	mobility = mobility[mobility['country_region'] == 'United States'].drop(
		columns=['country_region', 'country_region_code'])
	mobility.loc[mobility['sub_region_1'] == 'District of Columbia', \
		'sub_region_2'] = 'District of Columbia'  
	mobility = mobility[pd.notnull(mobility['sub_region_1']) & pd.notnull(
		mobility['sub_region_2'])]

	# Rename columns
	mobility = mobility.rename(columns={'sub_region_1': 'StateName',
	                                    'sub_region_2': 'CountyName'})

	# Google data uses out-of-date name for Oglala County, SD
	mobility.loc[(mobility['CountyName'] == 'Shannon County') \
	             & (mobility['StateName'] == 'South Dakota'), 'CountyName'] = 'Oglala Lakota County'

	# Change names to uppercase
	mobility['StateName'] = mobility['StateName'].apply(lambda x: x.upper())
	mobility['CountyName'] = mobility['CountyName'].apply(lambda x: x.upper())

	return mobility


def merge_mobility_fips(mobility, fips):

	# Merge mobility data onto FIPS codes (using county names)
	mobility_fips = pd.merge(mobility, fips, on=['StateName', 'CountyName'], how='left', indicator=True)
	assert((mobility_fips['_merge']=='both').all())
	mobility_fips = mobility_fips.drop(columns=['_merge'])

	return mobility_fips


def create_lag_features(df, varnames):

	df = df.copy(deep=True)

	print('Lagging...')
	df.set_index(['date', 'fips'], inplace=True)
	df.sort_index(inplace=True)

	for lag in [1, 3, 5, 7, 10]:
		print('\tLag num', lag)
		lagged = df[varnames].groupby(level='fips').shift(lag)
		lagged.columns = ['lag{}_{}'.format(lag, c) for c in lagged.columns]
		df = df.merge(lagged, how='left', left_index=True, right_index=True)

	return df.reset_index()


def create_moving_avgs(df):

	df = df.copy(deep=True)

	cols = ([
		'retail_and_recreation_percent_change_from_baseline',
		'grocery_and_pharmacy_percent_change_from_baseline',
		'parks_percent_change_from_baseline',
		'transit_stations_percent_change_from_baseline',
		'workplaces_percent_change_from_baseline',
		'residential_percent_change_from_baseline'
			])

	varnames = []

	# Rolling average
	df.set_index('date', inplace=True)
	df.sort_index(inplace=True)
	for var in cols:
		for window in [3, 7]:
			varname = 'chg_{}_{}d_avg'.format(var[:var.find('_')], window)
			varnames.append(varname)
			df[varname] = df.groupby(
				'fips')[var].transform(lambda x: x.rolling(window, 1).mean())

	return df.reset_index(), varnames


def interpolate_rolling(df):

	for c in df.columns:
		if 'avg' in c and 'chg' in c:
			print('imputing for', c)
			vals = df.groupby(['StateFIPS', 'date'])[c].transform(np.mean)
			df[c].fillna(vals, inplace=True)

	# df.drop('interpolate_val', axis=1, inplace=True, errors='raise')

	return df




# def main():

# 	path = join(RAW, 'Global Mobility.csv')
# 	mobility = pd.read_csv(join(path))

# 	### filter
# 	us_mob = filter_counties(mobility)


# 	county = pd.read_csv(join(RAW, 'Crosswalk - FIPS-CountyName.csv')).astype(object)
# 	county['CountyFIPS'] = county['CountyFIPS'].apply(lambda x: utils.prepend_0s(str(x), 3))
# 	county['StateFIPS'] = county['StateFIPS'].apply(lambda x: utils.prepend_0s(str(x), 2))

# 	merged = us_mob.merge(county, how='left', left_on=('county'), right_on='CountyName')

# 	merged.to_csv(join(CL, 'us_mobility.csv'), index=False)

# 	return merged


# def filter_counties(df):

# 	new_df = df[(df['country_region'] == 'United States') & df['sub_region_2'].notnull()]
# 	new_df.rename({'sub_region_2': 'county'}, axis=1, inplace=True)

# 	return new_df




# if __name__ == "__main__":

	# main()
