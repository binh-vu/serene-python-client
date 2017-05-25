#!/bin/bash

CHUFFED_PATH=/home/diegod1/WORK/phd-side-projects/chuffed/branches/chuffed-ddg/chuffed
MZN2FZN_EXEC=~/bin/minizinc-2.1.4/bin/mzn2fzn

echo "Converting all graphs to dzn."
PASSED=$1

if [ ! -d "${PASSED}" ] ; then
	echo "Argument is not valid. Usage: pass directory containing graphml files";
        exit 1
fi

if [ ! -f $1/alignment.graphml ] ; then
	echo "Argument is not valid. Usage: pass directory containing graphml files";
        exit 1
fi


echo "Generating alignment.dzn"
python graphml2nice.py $1/alignment.graphml > $1/alignment.dzn 
for i in $1/*integration.graphml
do
	f=${i##*/}
	echo "Generating" $1/${f%.*}.dzn
	python graphml2matching.py $i > $1/${f%.*}.dzn
done

read -p "Generate fzn's? [Y/n] " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
	for i in $1/*.integration.dzn
	do
		f=${i##*/} 
		echo "Generating " $1/${f%.*}.fzn "and" $1/${f%.*}.ozn "from" $i
		$MZN2FZN_EXEC -I $CHUFFED_PATH/globals/ model.mzn $1/alignment.dzn $i -o $1/${f%.*}.fzn --output-base $1/${f%.*}
	done
fi
