#!/bin/bash

SCRIPT_PATH="/scratch2/gloukatou/fair_project//wordseg_whole_acqdiv/" #path where scripts are
ACQDIV_PATH="/Users/lscpuser/Downloads/" #path where the acqdiv data are
CSV_PATH="/scratch2/gloukatou/fair_project/wordseg_whole_acqdiv/acqdiv_final_data/" #path with utterances.csv files extracted from ACQDIV
LANGUAGE="Japanese" #language, others: Sesotho, Yucatec, Russian, Inuktitut, Chintang, Turkish, Indonesian
RESULT_PATH="/scratch2/gloukatou/fair_project/wordseg_whole_acqdiv/final_data/" 

############################  user, you are done :)


#mkdir ${CSV_PATH}/${LANGUAGE} #must have a subfolder with name of language
#####extract csv from Robject with R script
#Rscript  ${SCRIPT_PATH}split_clean_phonemize_wordseg.R CSV_PATH ${ACQDIV_PATH}segmented_acqdiv_corpus_2018-08-27.Rdata ${ACQDIV_PATH}acqdiv_corpus_2018-08-27.rda LANGUAGE

#####convert .csv to .tsv and remove child speech
#python ${SCRIPT_PATH}prepareAcqdiv.py --language $LANGUAGE --corpusinit ${CSV_PATH} --corpusfinal ${CSV_PATH}

#####segment
bash ${SCRIPT_PATH}pickandsegment.sh $LANGUAGE ${CSV_PATH} ${RESULT_PATH} $SCRIPT_PATH

