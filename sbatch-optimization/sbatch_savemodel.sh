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

if [ $language = "Inuktitut" ]; then
 

batchsize=16
char_embedding_size=50
hidden_dim=1024
layer_num=1
weight_dropout_in=0.3
weight_dropout_hidden=0.5
char_dropout_prob=0.05
char_noise_prob=0
learning_rate=1
sequence_length=40
epoch=13

elif [ $language = "Sesotho" ]; then


batchsize=32
char_embedding_size=100
hidden_dim=512
layer_num=1
weight_dropout_in=0.3
weight_dropout_hidden=0.5
char_dropout_prob=0.05
char_noise_prob=0.0
learning_rate=10
sequence_length=40
epoch=4

elif [ $language = "Indonesian" ]; then


batchsize=32
char_embedding_size=50
hidden_dim=1024
layer_num=2
weight_dropout_in=0.05
weight_dropout_hidden=0.05
char_dropout_prob=0.05
char_noise_prob=0.0
learning_rate=1
sequence_length=40
epoch=38


elif [ $language = "Chintang" ]; then


batchsize=16
char_embedding_size=100
hidden_dim=512
layer_num=1
weight_dropout_in=0.3
weight_dropout_hidden=0.5
char_dropout_prob=0.05
char_noise_prob=0
learning_rate=2
sequence_length=40
epoch=18



elif [ $language = "Yucatec" ]; then


batchsize=16
char_embedding_size=100
hidden_dim=1024
layer_num=1
weight_dropout_in=0.05
weight_dropout_hidden=0.3
char_dropout_prob=0.05
char_noise_prob=0
learning_rate=1
sequence_length=40
epoch=20


elif [ $language = "Russian" ]; then


batchsize=32
char_embedding_size=50
hidden_dim=512
layer_num=1
weight_dropout_in=0.3
weight_dropout_hidden=0.3
char_dropout_prob=0.05
char_noise_prob=0
learning_rate=10
sequence_length=40
epoch=23


elif [ $language = "Turkish" ]; then


batchsize=16
char_embedding_size=100
hidden_dim=512
layer_num=2
weight_dropout_in=0.3
weight_dropout_hidden=0.5
char_dropout_prob=0.05
char_noise_prob=0
learning_rate=1
sequence_length=40
epoch=26



fi


echo $language

save_to="/scratch2/gloukatou/fair_project/checkpoint/"

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



#count=0
sbatch run_one_seg_savemodel.sh \
                                               " --language $language \
                                               --save-to $save_to \
                                               --epoch $epoch \
                                               --batchSize $batchsize \
                                               --char_embedding_size $char_embedding_size \
                                               --hidden_dim $hidden_dim \
                                               --layer_num $layer_num \
                                               --weight_dropout_in $weight_dropout_in \
                                               --weight_dropout_hidden $weight_dropout_hidden \
                                               --char_dropout_prob $char_dropout_prob \
                                               --char_noise_prob $char_noise_prob \
                                               --learning_rate $learning_rate \
                                               --sequence_length $sequence_length " 




#for batchsize in  32  
#do
 #   for char_embedding_size in 50 
  #  do
   #    for hidden_dim in  512
    #    do
     #       for layer_num
