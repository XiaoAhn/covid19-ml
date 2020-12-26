import load_cl_target
import load_cl_CDC
import import_health
import pull_votes
import pull_census
import load_interventions
import pull_noaa



def populate_intermediate_data():
	"""
	Populates intermediate data folder
	"""
	print("buckle up, you're in for a long ride...")

	### target
	print("cleaning target var...")
	load_cl_target.execute()

	### census sources
	print("pulling ACS data...")
	pull_census.execute_ACS()

	print("pulling Business Pattern data...")
	pull_census.execute_NAICS()

	### noaa data
	print("pulling weather data...")
	pull_noaa.main()

	### CDC cases and deaths
	print("pulling CDC cases and deaths...")
	load_cl_CDC.main()

	### CDC health characteristics
	print("pulling CDC health data...")
	import_health.execute()

	### MIT Election Lab
	print("pulling MIT Election Lab...")
	pull_votes.main()

	### Interventions
	print("pulling interventions...")
	load_interventions.execute()


if __name__ == '__main__':

	populate_intermediate_data()