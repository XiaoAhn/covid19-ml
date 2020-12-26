import pandas as pd
# import numpy as np
from os.path import join, abspath, dirname
import utils

ROOT = join(dirname(dirname(abspath(__file__))))
RAW = join(ROOT, 'data_raw')
INT = join(ROOT, 'data_intermediate')


def execute():
	"""
	runs steps to create intermediate data files for cdc health data
	many of the steps resolves unusual geographic levels of the data
	"""

	healthdf_msa = get_cbsa_level()	
	healthdf_st = get_state_level()
	healthdf_msa.to_csv(join(INT, 'healthdf_msa.csv'), index=False)
	healthdf_st.to_csv(join(INT, 'healthdf_st.csv'), index=False)

	cbsa_fips_crosswalk = get_cbsa_crosswalk()
	metrodiv_fips_crosswalk = get_metro_division()
	cbsa_fips_crosswalk.to_csv(join(INT, 'cbsa_fips.csv'), index=False)
	metrodiv_fips_crosswalk.to_csv(join(INT, 'metrodiv_fips.csv'), index=False)

	health_fips = merge_crosswalks(healthdf_msa, cbsa_fips_crosswalk, metrodiv_fips_crosswalk)
	health_fips.to_csv(join(INT, 'health_fips.csv'), index=False)



	return healthdf_st


def merge_health_dfs(healthdf_fips, healthdf_st):
	"""
	interpolates missing fips health data
	"""
	state_fips = pd.read_csv(join(RAW, "Crosswalk - FIPS-StateName.csv"))
	state_fips['StateFIPS'] = state_fips['StateFIPS'].apply(lambda x: utils.prepend_0s(str(x), 2))
	state_fips['StateName'] = state_fips['StateName'].apply(str.upper)

	healthdf_st = healthdf_st.merge(state_fips, how='inner', on='StateName')

	healthdf_fips['FIPS_ID'] = healthdf_fips['FIPS_ID'].apply(lambda x: utils.prepend_0s(str(x), 5))
	healthdf_fips['StateFIPS'] =  healthdf_fips['FIPS_ID'].str[:2]

	for c in healthdf_fips.columns:
		if c.startswith('Percent'):
			healthdf_fips.loc[healthdf_fips[c].isnull(), c] = \
				healthdf_fips[healthdf_fips[c].isnull()].apply(
				lambda x: interpolate_nulls(x['StateFIPS'], c, healthdf_st), axis=1)

	return healthdf_fips


def interpolate_nulls(null_state, col, healthdf_st):

	return healthdf_st.loc[healthdf_st['StateFIPS'] == null_state, col]




def get_cbsa_level():
	# Get health data and rename columns
	healthdf_msa = pd.read_csv(join(RAW, "BRFSS Health - CBSA.csv"))

	# Rename columns
	healthdf_msa = healthdf_msa.rename(columns={'Locationabbr':'CBSA_ID',
	                                            'Locationdesc':'CBSA_Name_BRFSS',
	                                            'Data_value':'Percent'})

	# Subset to just most recent data and general population
	healthdf_msa = healthdf_msa[healthdf_msa['Year'] == max(healthdf_msa['Year'])]
	healthdf_msa = healthdf_msa[healthdf_msa['Break_Out'] == 'Overall']

	# Subset to just questions we want
	questions_kept = pd.read_csv(join(RAW, "BRFSS Health - Questions To Keep.csv"))
	healthdf_msa = pd.merge(questions_kept, healthdf_msa, on='QuestionID', how='inner')

	# Subset to just response %'s we want (i.e., only keep 'Yes' response %)
	healthdf_msa = healthdf_msa[(healthdf_msa['Response'] == 'Yes') | \
	                            (healthdf_msa['Response'] == 'Obese (BMI 30.0 - 99.8)')]

	# Reshape wide 
	healthdf_msa = healthdf_msa[['CBSA_ID','CBSA_Name_BRFSS', 'Percent', 'QuestionDesc']]
	healthdf_msa = healthdf_msa.set_index(['CBSA_ID', 'CBSA_Name_BRFSS', 'QuestionDesc']) \
	                           .unstack() \
	                           .reset_index()
	healthdf_msa.columns = [col[0] + col[1] for col in healthdf_msa.columns]

	# Change CBSA to string and change 1 column name
	healthdf_msa['CBSA_ID'] = healthdf_msa['CBSA_ID'].astype(str)
	healthdf_msa = healthdf_msa.rename(columns={'PercentBMI': 'PercentObese'})

	return healthdf_msa


def get_state_level():

	# Get health data and rename columns
	healthdf_st = pd.read_csv(join(RAW, "BRFSS Health - State.csv"))

	# Rename columns
	healthdf_st = healthdf_st.rename(columns={'Locationabbr':'StateID',
	                                          'Locationdesc':'StateName',
	                                          'Data_value':'Percent'})

	# Subset to just most recent data and general population
	healthdf_st = healthdf_st[healthdf_st['Year']==max(healthdf_st['Year'])]
	healthdf_st = healthdf_st[healthdf_st['Break_Out']=='Overall']

	# Subset to just questions we want
	questions_kept = pd.read_csv(join(RAW, "BRFSS Health - Questions To Keep.csv"))
	healthdf_st = pd.merge(questions_kept, healthdf_st, on='QuestionID', how='inner')

	# Subset to just response %'s we want (i.e., only keep 'Yes' response %)
	healthdf_st = healthdf_st[(healthdf_st['Response'] == 'Yes') | \
	                          (healthdf_st['Response'] == 'Obese (BMI 30.0 - 99.8)')]

	# Reshape wide 
	healthdf_st = healthdf_st[['StateID','StateName','Percent','QuestionDesc']]
	healthdf_st = healthdf_st.set_index(['StateID','StateName','QuestionDesc']) \
	                         .unstack() \
	                         .reset_index()
	healthdf_st.columns = [col[0]+col[1] for col in healthdf_st.columns]

	# Change 1 column name
	healthdf_st = healthdf_st.rename(columns={'PercentBMI':'PercentObese'})
	healthdf_st.loc[:, 'StateName'] = healthdf_st['StateName'].str.upper()

	#### merge on fips
	state_fips = pd.read_csv(join(RAW, "Crosswalk - FIPS-StateName.csv"))
	state_fips['StateFIPS'] = state_fips['StateFIPS'].apply(lambda x: utils.prepend_0s(str(x), 2))
	state_fips['StateName'] = state_fips['StateName'].apply(str.upper)

	healthdf_st = healthdf_st.merge(state_fips, how='inner', on='StateName')

	return healthdf_st


def get_cbsa_crosswalk():

	# Get first crosswalk
	cbsa_fips_crosswalk = pd.read_csv(join(RAW, "Crosswalk - CBSA-FIPS.csv"))

	# Fix FIPS code with 0 at beginning
	cbsa_fips_crosswalk['CBSA_ID'] = cbsa_fips_crosswalk['CBSA_ID'].astype(str)
	cbsa_fips_crosswalk['FIPS_ID'] = cbsa_fips_crosswalk['FIPS_ID'].astype(str)
	cbsa_fips_crosswalk['FIPS_ID'] = cbsa_fips_crosswalk['FIPS_ID'].apply(
		lambda x: utils.prepend_0s(str(x), 5))

	# Two manual fixes
	cbsa_fips_crosswalk.loc[cbsa_fips_crosswalk['CBSA_ID'] == '30100', 'CBSA_ID'] = '17200'
	cbsa_fips_crosswalk.loc[cbsa_fips_crosswalk['CBSA_ID'] == '19430', 'CBSA_ID'] = '19380'

	return cbsa_fips_crosswalk


def get_metro_division():

	# Get metro division crosswalk
	metrodiv_fips_crosswalk = pd.read_csv(join(RAW, "Crosswalk - MetroDiv-FIPS.csv"))

	# Keep only desired columns
	metrodiv_fips_crosswalk = metrodiv_fips_crosswalk[['STATEFP', 'COUNTYFP', \
								'NAMELSAD', 'METDIVFP', 'METRO_DIVISION']]
	metrodiv_fips_crosswalk = metrodiv_fips_crosswalk[pd.notnull(
		metrodiv_fips_crosswalk['METDIVFP'])]

	# Rename columns
	metrodiv_fips_crosswalk = metrodiv_fips_crosswalk.rename(
		columns={'NAMELSAD': 'FIPS_Name', 'METDIVFP': 'CBSA_ID', 'METRO_DIVISION': 'CBSA_Name'})

	# Fix FIPS with wrong state FIPS digit numbers
	metrodiv_fips_crosswalk['CBSA_ID'] = metrodiv_fips_crosswalk['CBSA_ID'].astype(int).astype(str)
	metrodiv_fips_crosswalk['STATEFP'] = metrodiv_fips_crosswalk['STATEFP'].astype(str)
	metrodiv_fips_crosswalk['STATEFP'] = metrodiv_fips_crosswalk['STATEFP'].apply(
		lambda x: utils.prepend_0s(str(x), 2))

	# Fix FIPS with wrong county FIPS digit numbers
	metrodiv_fips_crosswalk['COUNTYFP'] = metrodiv_fips_crosswalk['COUNTYFP'].astype(str)
	metrodiv_fips_crosswalk['COUNTYFP'] = metrodiv_fips_crosswalk['COUNTYFP'].apply(
		lambda x: utils.prepend_0s(str(x), 3))

	# Get combined FIPS code
	metrodiv_fips_crosswalk['FIPS_ID'] = metrodiv_fips_crosswalk['STATEFP'] + \
		metrodiv_fips_crosswalk['COUNTYFP']
	metrodiv_fips_crosswalk = metrodiv_fips_crosswalk.drop(
		columns=['STATEFP', 'COUNTYFP'])

	return metrodiv_fips_crosswalk


def merge_crosswalks(healthdf_msa, cbsa_fips_crosswalk, metrodiv_fips_crosswalk):

	# Merge CBSA-health data onto FIPS codes (using CBSA / MetroDiv IDs)
	merged1 = pd.merge(healthdf_msa, cbsa_fips_crosswalk, on='CBSA_ID', how='inner')
	merged2 = pd.merge(healthdf_msa, metrodiv_fips_crosswalk, on='CBSA_ID', how='inner')
	health_fips = pd.concat([merged1, merged2], ignore_index=True, sort=False)

	return health_fips