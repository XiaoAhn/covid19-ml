import pandas as pd
import numpy as np
from os.path import join, abspath, dirname

ROOT = join(dirname(dirname(abspath(__file__))))
INT = join(ROOT, 'data_intermediate')
RAW = join(ROOT, 'data_raw')


def main():
	"""
	executes steps to make intermediate data CSV for MIT election data
	there must be a file in the data_raw folder titled 'countypres_2000-2016.csv'
	includes calculating vote share from columns
	"""

	votes = pd.read_csv(join(RAW, 'countypres_2000-2016.csv'))
	votes = votes.loc[(votes['year'] == 2016) & votes['party'].notnull(), (
		['FIPS', 'party', 'totalvotes', 'candidatevotes'])]
	votes['vote_share'] = votes['candidatevotes'] / votes['totalvotes']
	votes.drop(['candidatevotes'], axis=1, inplace=True, errors='raise')
	df = votes.pivot_table(
		index='FIPS', columns=['party'], values=['vote_share']).reset_index()
	df.columns = ['fips', 'voteshare_dem', 'voteshare_rep']
	df = df.merge(votes[['FIPS', 'totalvotes']].drop_duplicates(ignore_index=True),
		how='inner', left_on='fips', right_on='FIPS').drop('FIPS', axis=1)
	df.to_csv(join(INT, 'votes.csv'), index=False)

	return df