#! /bin/bash

max_index=50
index=0

while [ $index -lt $max_index ]
do
    echo $index
    echo $position
    ./exe_mcmpc 
    # ./exe_mcmpc 10
    # ./exe_mcmpc echo "scale=2; $tics / 100" | bc | xargs printf "%.2f\n"
    # position= echo $((position + 0.01))
    index=`expr $index + 1`
    # tics=`expr $tics + 1`
    # position="scale=2; $tics / 100" | bc | xargs printf "%.2f\n"
    mv output/data_iteration_*.txt Geometric-40/data_iteration_"$index".txt
    cd output/
    rm *.txt
    cd ../
done
