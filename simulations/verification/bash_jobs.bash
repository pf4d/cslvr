#!/bin/bash

# first, get the maximum number of processors :
max_proc=$(cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l);
echo "maximum number of processors = $max_proc";

# function returns the number of vertices generated in `ismip_hom_a.py' :
function v()
{
	local v_i=$(( ($1+1) * ($1+1) * ($1/4 + 1) ));
	echo $v_i;
}

# run the verification for increasing dofs :
for ((i=10; i<=100; i++)); do
	v_i=$(v $i);

	# set the number of processes so that no more than 1000 vertices are
	# distributed per processor :
	n_p=$(( $v_i / 1000 + 1 ));

	# break if we can't increase the number of processes :
	if [ $n_p -gt $max_proc ]; then
		break;
	fi

	# otherwise, run the program :
	echo -e "i = $i \t v = $v_i \t n_p = $n_p";
	echo -e "\t executing : mpirun -np $n_p python ismip_hom_a.py $i";
	mpirun -np $n_p python ismip_hom_a.py $i;
done

