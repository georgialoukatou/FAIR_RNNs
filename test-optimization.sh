#!/usr/bin/env bash

weight_dropout_in_range=()
weight_dropout_hidden_range=()
char_dropout_prob_range=()

language=$1


i=0
for x in {1..8}; do
num=$(echo "$x / 20.0" | bc -l)
weight_dropout_in_range+=("$num") ;done

y=0
for y in {1..10}; do
num=$(echo "$y / 20.0" | bc -l)
weight_dropout_hidden_range+=("$num");done

z=0
for z in {1..15}; do
num=$(echo "$z / 40.0" | bc -l)
char_dropout_prob_range+=("$num");done

for batchsize in  1 2 3 ;
        do
        for char_embedding_size in 50;
                do
                for hidden_dim in 3 2;
                        do
                        for layer_num in 1 2 3;
                                do
                                for weight_dropout_in in "${weight_dropout_in_range[@]}"
                                        do
                                        echo "$weight_dropout_in"
                            #            for weight_dropout_hidden in "${weight_dropout_hidden_range[@]}"
                             #                   do
                              #                  for char_dropout_prob in "${char_dropout_prob_range[@]}"
                               #                         do
                                #                        for char_noise_prob in 0.0 0.01 0.02 0.05;
                                 #                               do
                                  #                              for learning_rate in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8;
                                   #                                     do
                                    #                                    for sequence_length in 30 40 50 80;
                                     #                                           do
#       python lm-acqdiv.py --language $language --batchSize $batchsize --char_embedding_size $char_embedding_size --hidden_dim $hidden_dim --layer_num $layer_num --weight_dropout_in $weight_dropout_in --weight_dropout_hidden $weight_dropout_hidden --char_dropout_prob $char_dropout_prob --char_noise_prob $char_noise_prob --learning_rate $learning_rate --sequence_length $sequence_length  --save-to acqdiv-indonesian-initial
                                          #                                      done
                                         #                               done
                                        #                        done
                                       #                 done
                                      #          done
                                     #  done
                                done
                        done
                done
	done
done



optimal_values=$(python min_loss.py 2>&1)
echo "Epoch MinDevloss = ${optimal_values##*)}, Optimal Parameters = ${optimal_values%)*}"
