#!/usr/bin/env bash

# Selecting speakers from cha files in prep to generating a csv format.
#
# IMPORTANT!! Includes data selection.
#


######### VARIABLES
# Variables that have been passed by the user
RAW_FOLDER="/Users/lscpuser/Documents/lstm_acquisition_Typology/"
output_file=$RAW_FOLDER/output.txt
output_file1=$RAW_FOLDER/output1.txt
output_file2=$RAW_FOLDER/output2.txt
output_file3=$RAW_FOLDER/output3.txt
output_file4=$RAW_FOLDER/output4.txt


for corpus in $(find $RAW_FOLDER -type d);
 do
 [ -d "$corpus" ]; 
 for chafile in $corpus/*cha;
   do
    cat $chafile | tr -d '\n'| tr '\*' '\n' | tr -d '[:digit:]' | sed 's/([^)]*)//g;s/  / /g'  |  sed 's/\[.*\]//g' | sed 's/  / /g' >> $output_file    
   done
done


grep  -v  '^@\|^CHI\|^ .*$'   $output_file >> $output_file1
cat $output_file1 | tr '\_'  '\n'|   tr '[:upper:]' '[:lower:]'  > $output_file2

grep  -v  '\%xmor\|\.\.\.\|xxx'   $output_file2 >> $output_file3
cat $output_file3 |sed "s/\'//g" |  sed "s/\"//g" |sed 's/[\^;-_%]//g'  | sed 's/@//g'| sed 's/+< //g' | sed 's/\+//g' | sed 's/<//g' | sed 's/\<.*\>//g'  | sed 's/[^ ]*\|[^ ]*//g' |sed 's/,//g' | sed 's/\///g' | sed 's/&[^[:space:]]*//g'   |  sed 's/\!//g' | sed 's/\?//g' | sed 's/\.//g' |sed 's/,//g'  | grep  ':' | sed 's/„//g' | sed 's/	/ /g' | sed 's/:://g' |  sed 's/^ //g' | sed '/: *$/d'  |  tr ':' '\t' | sed 's/	  /	 /g' > $output_file4


