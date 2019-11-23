import sys
import os
language = sys.argv[1]

files = [x for x in os.listdir("tuning_results/") if x.startswith(f"{language}_lm-acqdiv-split-new.py_")]

results = []
with open(f"tuning_results/results_{language}_lm-acqdiv-split-new.py.txt", "w") as outFile:
 for name in files:
   with open(f"tuning_results/{name}", "r") as inFile:
      args, losses = inFile.read().strip().split("\n")
      loss = min([float(x) for x in losses.split(" ")])
      args = args[10:-1].replace(", ", " ")
      results.append((loss, args))
 results = sorted(results)
 for r in results:
      print("\t".join([str(x) for x in r]), file=outFile)


