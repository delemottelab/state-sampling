wdir=$1
view=$2
feature_type=$3
classifier=$4
traj_type=$5
if [ $# -lt 6 ]
  then
	infilename="importance"
else
	infilename="$6_importance"
fi
importance_file="$wdir/${traj_type}/${feature_type}/$classifier/$infilename.pdb"
if [ ! -f "${importance_file}" ] ; then
	echo "File not found: ${importance_file}"
    exit 1
fi
echo $importance_file
cmd="vmd -dispdev none -e $wdir/${view}view_vmd_template.tcl -args ${importance_file} $wdir/structures/${view}_${feature_type}_${classifier}_${traj_type}_${infilename}"
echo $cmd
$cmd
echo ""
echo "Done"