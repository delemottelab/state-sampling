#!/usr/bin/env bash
ligand=$1
iter=$2
exploration=$3
if [[ $exploration == *"multi"* ]]; then
  wd=".simu/multistate"
else
  wd=".simu/singlestate"
fi
cmd="python main.py --iteration=$iter --working_dir=$wd/$ligand/ --exploration_type=$exploration --max_iteration=8"
echo $cmd
$cmd >> $ligand-$exploration-server.log 2>&1 &
