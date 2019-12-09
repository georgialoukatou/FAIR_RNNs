

text = open("/Users/lscpuser/Documents/lstm_acquisition_Typology/output5.csv", 'r')
morphLine=[]
uttLine=[]

count = 0
count1= 1
for line in text:	
        count+=1
        count1+=1
        if count % 2 == 0: #this is the remainder operator
            morphLine.append(line)
        if count1 % 2 == 0: #this is the remainder operator
           uttLine.append(line)
text.close()            

#print(morphLine[0])
#print(uttLine[0])

dictionary=[]
for utt, mor in zip(uttLine, morphLine):
    utt=utt.split("            ") 
    mor=mor.split("            ")
    for utt_,  mor_ in zip(utt,mor):
  #   print(utt_, mor_)
     dictionary.append([utt_, mor_])

for element  in dictionary:
    for item in element:
     if "V|" in item and "ADV" not in item:
      print(element)     
#print(dictionary[0:10])