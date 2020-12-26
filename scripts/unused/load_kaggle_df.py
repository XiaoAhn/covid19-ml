import pandas as pd
from os.path import join, abspath, dirname

ROOT = join(dirname(dirname(abspath(__file__))))
OUTPATH = join(ROOT, 'data_intermediate')

cols = ([
		'date',
		'fips',
		'area_sqmi',
		'stay_at_home_announced',
		'stay_at_home_effective',
	])
# cols = ([
# 		'date',
# 		'county',
# 		'state',
# 		'fips',
# 		'area_sqmi',
# 		'stay_at_home_announced',
# 		'stay_at_home_effective',
# 		'mean_temp',
# 		'wind_speed',
# 		'precipitation'
# 	])


def main():
	df = read_data()
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	# df = interpolate_NAs(df[cols])
	# df.set_index('date', inplace=True)
	# df.sort_index(inplace=True)
	df = df[cols]
	df = create_features(df)
	df.to_csv(join(OUTPATH, 'cl_kaggle.csv'))
	return df


def read_data():

	df = pd.read_csv(join(ROOT, 'data_raw',
		'US_counties_COVID19_health_weather_data.csv'))
	return df


def create_features(df):

	# new_df = df[cols]
	new_df = df.copy(deep=True)
	# new_df['precip_dummy'] = 0
	# new_df.loc[new_df['precipitation'] > .05, 'precip_dummy'] = 1 ## arbitrary cutoff for low precipitation

	# new_df['mean_tmp_3d_avg'] = df.groupby('fips')['mean_temp'].transform(
	# 	lambda x: x.rolling(3, 1).mean())

	new_df[['stay_at_home_announced', 'stay_at_home_effective']] = new_df[['stay_at_home_announced', 'stay_at_home_effective']].replace({'no': 0, 'yes': 1})

	return new_df


# def interpolate_NAs(df):

# 	states = df.groupby(['state', 'date'])['mean_temp', 'wind_speed', 'precipitation'].mean()
# 	states = states.rename({c: c + "_state" for c in states.columns},
# 		axis=1, errors='raise').reset_index()
# 	new_df = df.merge(states, how='left', on=['state', 'date'])
# 	for c in ['mean_temp', 'wind_speed', 'precipitation']:
# 		new_df.loc[new_df[c].isnull(), c] = new_df.loc[new_df[c].isnull(), c + '_state']

# 	new_df.drop([c for c in new_df.columns if c.endswith('_state')], 
# 		axis=1, inplace=True, errors='raise')

# 	return new_df


