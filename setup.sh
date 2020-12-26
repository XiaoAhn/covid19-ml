#!/bin/bash 



# This file sets up data folders on the repo to have the necesary
# files to run .py scrips.  It requires using the zip and unzip
# commands for linux.  This commands will be installed if they don't
# already exist.





echo "Checking Python version..."
REQ_PYTHON_V="360"

ACTUAL_PYTHON_V=$(python -c 'import sys; version=sys.version_info[:3]; print("{0}{1}{2}".format(*version))')
ACTUAL_PYTHON3_V=$(python3 -c 'import sys; version=sys.version_info[:3]; print("{0}{1}{2}".format(*version))')

if [[ $ACTUAL_PYTHON_V > $REQ_PYTHON_V ]] || [[ $ACTUAL_PYTHON_V == $REQ_PYTHON_V ]];  then 
    PYTHON="python"
elif [[ $ACTUAL_PYTHON3_V > $REQ_PYTHON_V ]] || [[ $ACTUAL_PYTHON3_V == $REQ_PYTHON_V ]]; then 
    PYTHON="python3"
else
    echo -e "\tPython 3.7 is not installed on this machine. Please install Python 3.7 before continuing."
    exit 1
fi

echo -e "\t--Python 3.7 is installed"

echo "can we create a virtual environment on your machine[Y|n]"
while [[ "$resp" != "Y" && "$resp" != "n" ]]; do
	read resp
	echo
done

if [[ "$resp" == "n" ]]; then
	echo "you said no... exiting..."
	exit 1
fi

# Remove the env directory if it exists 
if [[ -d env ]]; then 
    rm -r env  
fi

echo -e "Creating virtual environment..."
$PYTHON -m venv env 
if [[ ! -d env ]]; then 
    echo -e "\t--Could not create virutal environment... Please make sure venv is installed"
    exit 1
fi

# 3. Install requirements 

echo -e "Installing Requirements"
if [[ ! -e "requirements.txt" ]]; then 
    echo -e "\t--Need requirements.txt to install packages."
    exit 1
fi

source env/bin/activate
pip install -r requirements.txt || run_extra_installs




########## setting up folders ##############33

echo "To unzip the files we first have to check that you have zip and unzip installed"
echo "Can you give us permission to install zip and unzip if they're not found? [Y|n]"

while [[ "$resp" != "Y" && "$resp" != "n" ]]; do
	read resp
	echo
done

if [[ "$resp" == "n" ]]; then
	echo "you said no... feel free to manually unzip all files"
	exit 1
fi


for pkg in zip unzip; do
	PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $pkg|grep "install ok installed")
	echo Checking for $pkg: $PKG_OK
	if [ "" = "$PKG_OK" ]; then
	  echo "No $pkg. Setting up $REQUIRED_PKG."
	  sudo apt-get --yes install $pkg 
	fi
done;


echo
echo "Unzipping files data_raw..."
unzip data_raw/data.zip -d data_raw
unzip data_raw/noaa.zip -d data_raw
unzip data_raw/shape.zip -d data_raw

echo
echo "Unzipping files data_intermediate"
unzip data_intermediate/data_intermediate.zip -d data_intermediate

echo -e "Done."


run_extra_installs() {

m="Pip installation of requirements failed.\nWould you like us to run apt-get commmands that resolved the issue on our machine? [Y|n]"

while [[ "$resp" != "Y" && "$resp" != "n" ]]; do

	echo -e $m
	read resp
	echo
  
done

if [[ "$resp" == "Y" ]]; then

	sudo apt-get update
	sudo apt install udo
	sudo apt-get install libspatialindex-dev
	sudo apt-get install -y python-rtree
fi

pip install -r requirements.txt

	exit 1
}
