with open("tuning_results/results_portuguese_lm-acqdiv-split-new.py.txt", "r") as inFile:
   ce, params = next(inFile).strip().split("\t")
   params=["--"+x for x in params.split(" ") if not x.startswith("myID=") and not x.startswith("save_to") and not x.startswith("out_loss_filename") and not x.startswith("load_from")]
import random

myID = random.randint(1000,100000000)
language = "Portuguese"
script = "lm-acqdiv-split-new.py"
args = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", script] + params + ["--myID="+str(myID), "--save_to="+language+"_"+script+"_"+str(myID)]
print(args)

import subprocess
subprocess.run(args)


