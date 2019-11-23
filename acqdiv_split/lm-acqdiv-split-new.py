# -*- coding: utf-8 -*-

from config import VOCAB_HOME, CHECKPOINT_HOME

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="portuguese")
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)
parser.add_argument("--out-loss-filename", dest="out_loss_filename", type=str)


import random

parser.add_argument("--batchSize", type=int, default=random.choice([4, 8,16,32,64])) # 1, 2, 4, 
parser.add_argument("--char_embedding_size", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=random.choice([256,512,1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([1,2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0, 0.0, 0.01]))
parser.add_argument("--learning_rate", type = float, default=random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=50)


args=parser.parse_args()
print(args)


from acqdivReadersplit_New import AcqdivReader

acqdivCorpusReader = AcqdivReader(args.language)



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

speakers = {}
characters = {}
count = 0
for snippet in acqdivCorpusReader.train:
 characters["NewSnippet"] = characters.get("NewSnippet", 0)+1

 for line in snippet:
   count += 1
   if count % 100 == 0:
     print(count)
   utterance = line[2].strip("\x15").strip()
   speaker = line[1]
#   print(speaker, list(utterance))
   speakers[speaker] = speakers.get(speaker,0)+1
   for character in utterance+"\n":
      characters[character] = characters.get(character, 0)+1
   
children = {"maf", "ami", "pri"}
print(characters)
print(speakers)
characters = sorted(list(characters.items()), key=lambda x:(-x[1], x[0]))
print(characters)
itos = [x[0] for x in characters]
stoi = dict(zip(itos, range(len(itos))))
print(stoi)



import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop


rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)


rnn_drop = rnn #WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)


train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
          yield param

parameters_cached = [x for x in parameters()]

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
   checkpoint = torch.load(CHECKPOINT_HOME+args.load_from+".pth.tar")
   for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)])

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
   numeric = []
   count = 0
   print("Prepare chunks")

   for snippet in data:
    numeric.append(stoi["NewSnippet"]+3)
    for line in snippet:
      utterance = line[2].strip("\x15").strip()
      speaker = line[1]
      if speaker in children:
          continue
      for character in utterance+"\n":
         if character == " ":
            continue
         numeric.append(stoi[character]+3 if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric[:args.sequence_length+1]
            numeric = numeric[args.sequence_length+1:]
   


bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * args.char_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())



def forward(numeric, train=True, printHere=False):
    input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
    target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)

    #  print(char_embeddings)
    #if train and (embedding_full_dropout_prob is not None):
    #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
    #else:
    embedded = char_embeddings(input_tensor)

    if train:
       embedded = char_dropout(embedded)
       mask = bernoulli_input.sample()
       mask = mask.view(1, args.batchSize, args.char_embedding_size)
       embedded = embedded * mask

    out, _ = rnn_drop(embedded, None)
    #      if train:
    #          out = dropout(out)


    if train:
      mask = bernoulli_output.sample()
      mask = mask.view(1, args.batchSize, args.hidden_dim)
      out = out * mask




    logits = output(out)
    log_probs = logsoftmax(logits)
    #   print(logits)
    #    print(log_probs)
    #     print(target_tensor)

    loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

    if printHere:
       lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(args.sequence_length, len(numeric))
       losses = lossTensor.data.cpu().numpy()
       for i in range((args.sequence_length-1)-1):
          print((losses[i][0], itos[numeric[0][i+1]-3]))
    return loss, len(numeric) * args.sequence_length

def backward(loss, printHere):
   optim.zero_grad()
   if printHere:
      print(loss)
   loss.backward()
   torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
   optim.step()

import time

devLosses = []
for epoch in range(200):
   print(epoch)
   print("Got data")
   training_chars = prepareDatasetChunks(acqdivCorpusReader.train, train=True)

   rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   while True:
      counter += 1
      try:
         numeric = [next(training_chars) for _ in range(args.batchSize)]
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      trainChars += charCounts
      if printHere:
         print((epoch,counter))
         print("Dev losses")
         print(devLosses)
         print("Chars per sec "+str(trainChars/(time.time()-startTime)))
      if counter % 20000 == 0 and epoch == 0:
        if args.save_to is not None:
           torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), CHECKPOINT_HOME+args.save_to+".pth.tar")


   rnn_drop.train(False)


   dev_data = acqdivCorpusReader.dev
   print("Got data")
   dev_chars = prepareDatasetChunks(dev_data, train=True)



   dev_loss = 0
   dev_char_count = 0
   counter = 0

   while True:
      counter += 1
      try:
          numeric = [next(dev_chars) for _ in range(args.batchSize)]
      except StopIteration:
          break
      printHere = (counter % 50 == 0)
      loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
      dev_loss += numberOfCharacters * loss.cpu().data.numpy()
      dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   with open("tuning_results/"+args.language+"_"+os.path.basename(__file__)+"_"+str(args.myID), "w") as outFile:
      print(args, file=outFile)
      outFile.write(" ".join([str(x) for x in devLosses]) + '\n')
   if  len(devLosses)>1 and devLosses[-2] < devLosses [-1]:
      min_loss="minimum loss=" + str(float(devLosses[-2])) +" epoch=" + str(epoch-1) + " args=" + str(args)
      break
   else:
      min_loss="minimum loss=" + str(float(devLosses[-1])) +" epoch=" + str(epoch) + " args=" + str(args)
 
   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), CHECKPOINT_HOME+args.save_to+".pth.tar")


print(min_loss)

if args.out_loss_filename is not None:
   with open(args.out_loss_filename, 'w') as out_loss_file:
          out_loss_file.write(min_loss + '\n')



	

