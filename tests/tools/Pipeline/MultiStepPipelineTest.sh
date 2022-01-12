#!/bin/bash

set -e
set -o pipefail

if [ "$#" -ne 6  ]; then
	echo "Illegal number of parameters" > /dev/stderr
	exit
fi

revng-pipeline -P=$1 Strings3:Root:Root -i FirstStep:Strings1:$2 -o End:Strings3:$3 -p $4 --load $5 -s
diff $3 $6
revng-pipeline -P=$1 Strings3:Root:Root -i FirstStep:Strings1:$2 -o End:Strings3:$3 -p $4 --load $5 -s
diff $3 $6
rm $4/SecondStep/Strings1
revng-pipeline -P=$1 Strings3:Root:Root -i FirstStep:Strings1:$2 -o End:Strings3:$3 -p $4 --load $5 -s
diff $3 $6
rm $3
revng-pipeline -P=$1 Strings3:Root:Root -i FirstStep:Strings1:$2 -o End:Strings3:$3 -p $4 --load $5 -s
diff $3 $6
