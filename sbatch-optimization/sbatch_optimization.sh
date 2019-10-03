#!/bin/bash
#
# Grid search for lm-acqdiv. Run multiple jobs using SLURM sbatch,
# each job scheduled separatly.
#
# Logged as 'gloukatou', run this script like:
#     ./sbatch_optimization.sh Indonesian
#     # use squeue and wait all the jobs are terminated
#     optimal_values=$(python ../min_loss.py 2>&1)
#     echo "Epoch MinDevloss = ${optimal_values##*)}, Optimal Parameters = ${optimal_values%)*}"


language=$1
if [ -z $language ]
then
    echo "language not specified, exiting"
    exit
fi
language_lower=$(echo $language | tr '[:upper:]' '[:lower:]')


#weight_dropout_in_range=()
#i=0
#for x in {1..8}
#do
 #   num=$(echo "$x / 20.0" | bc -l)
  #  weight_dropout_in_range+=("$num")
#done



count=0
for batchsize in  32  
do
    for char_embedding_size in 100   50 
    do
       for hidden_dim in 1024  512  
        do
            for layer_num in 2  1
            do
                for weight_dropout_in in  0.3 0.5   
                do
                    for weight_dropout_hidden in  0.5 0.3 0.05 
                    do
                        for char_dropout_prob in 0.05 
                        do
                            for char_noise_prob in 0.0
                            do
                                for learning_rate in 1  #0.05done  #1 #10done 
                                do
                                    for sequence_length in 50
                                    do
                                        sbatch run_one.sh \
                                               " --language $language \
						 --batchSize $batchsize \
						--char_embedding_size $char_embedding_size \
						--hidden_dim $hidden_dim \
						--layer_num $layer_num \
						--weight_dropout_in $weight_dropout_in \
						--weight_dropout_hidden $weight_dropout_hidden \
						--char_dropout_prob $char_dropout_prob \
						--char_noise_prob $char_noise_prob \
						--learning_rate $learning_rate \
						--sequence_length $sequence_length \
						--save-to acqdiv-$language_lower-initial"

                                        # # for testing just run 3 jobs (2 should be launched
                                        # # immediatly and the 3rd waiting)
                                        # count=$(( $count + 1 ))
                                        # [ $count -eq 3 ] && exit
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


# optimal_values=$($python min_loss.py 2>&1)
# echo "Epoch MinDevloss = ${optimal_values##*)}, Optimal Parameters = ${optimal_values%)*}"



                       # --hidden_dim $hidden_dim \
                        #                       --layer_num $layer_num \
                         #                      --weight_dropout_in $weight_dropout_in \
                          #                     --weight_dropout_hidden $weight_dropout_hidden \
                            #                   --char_dropout_prob $char_dropout_prob \
                           # #                   --char_noise_prob $char_noise_prob \
                              #                 --learning_rate $learning_rate \
                               #                --sequence_length $sequence_length \
