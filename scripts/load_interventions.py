import pandas as pd
from os.path import join, dirname, abspath
import utils
import datetime

ROOT = join(dirname(dirname(abspath(__file__))))
INT = join(ROOT, 'data_intermediate')
RAW = join(ROOT, 'data_raw')

file = 'https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/data/interventions.csv'

def execute():
	"""
	basic cleaning and transformations for interventions data
	"""

	df = pd.read_csv(file)

	df.drop(['stay at home', '>50 gatherings', '>500 gatherings', 'entertainment/gym'],
		axis=1, inplace=True, errors='raise')

	names = {c: 'int_date_' + c for c in df.columns if c not in ['FIPS', 'STATE', 'AREA_NAME']}
	df.rename(names, axis=1, inplace=True, errors='raise')

	df.to_csv(join(INT, 'interventions.csv'), index=False)

	return df


def transform_dates(df, c):
	"""
	transforms columns that contain ordinal date into a dummy
	where the value is 0 if its before the date and 1 if its after
	"""

	df[c].fillna(800000, inplace=True) ### arbitrary high date
	df[c] = df[c].apply(lambda x: datetime.date.fromordinal(int(x)))
	df['int_' + c] = 0
	df.loc[df[c] >= df['date'], 'int_' + c] = 1

	return df



