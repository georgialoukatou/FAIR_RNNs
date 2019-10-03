# -*- coding: utf-8 -*-

from config import ACQDIV_HOME
from config import VOCAB_HOME

import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)


args=parser.parse_args()
print(args)

basePath = ACQDIV_HOME+"/tsv/acqdiv_final_data/"+ args.language.lower()

textfile = basePath+"/utterances_train.tsv"
charfile = VOCAB_HOME + args.languagelower + "-char.txt"

#replacements = {'ʰ':'h', 'ː':'(:)','ʱ':'(hv)', 'ã':'a~', 'õ' : 'o~', 'ɨ̃':'ix~',  'ĩ':'i~', '':'', 'ũ':'u~', 'ɲ':'n~'}
replacements = {'ʈʰː':'T', 'lʰ':'L', 'dz':'D', 'nː':'N', 'dʰ̃':'D', 'mː':'M', 'dzʱ̃':'Z', 'sː':'S', 'lː':'1', 'tsʰː':'S'}

lines = []
with open(textfile) as infile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        lines.append(line)
with open(textfile, 'w') as outfile:
    for line in lines:
        outfile.write(line)

with open(charfile) as infile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        lines.append(line)
with open(charfile, 'w') as outfile:
    for line in lines:
        outfile.write(line)
