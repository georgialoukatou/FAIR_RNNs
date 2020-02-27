import pyconll

portuguese="/Users/admin/Downloads/Universal Dependencies 2.5/ud-treebanks-v2.5/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu"
testset= open("/Users/admin/Desktop/testset.txt", "w")

corpus = pyconll.load_from_file(portuguese)

nouns={}
adjectivesverbs={}
aux_lemmas = set()
for sentence in corpus:
    for token in sentence:
        if token.upos == 'NOUN' and token.deprel == 'nsubj':
            info_=':' + str(token.form).lower() + ':' + str(token.feats)
            if token.lemma not in nouns :
                nouns[token.lemma.lower()]=[info_]
            else:
                nouns[token.lemma.lower()]=[nouns[token.lemma.lower()], info_]
        if token.upos == 'VERB' or token.upos == 'ADJ':
            adjectivesverbs[token.lemma.lower()]=["test"]
  
    
for key,value in adjectivesverbs.items():
    if key in nouns:
    	del nouns[key]


for keys,values in nouns.items():
    if "'Fem'" in str(values) and "'Masc'" not in str(values):
        token=str(values).split(':')[1]
        if "'Sing'" in str(values):
         testset.write("T:a " + str(token) + '\n')
         testset.write("F:o "+ str(token)+ '\n')
        elif "'Plur'" in str(values):
         testset.write("T:as "+ str(token)+ '\n')
         testset.write("F:os "+ str(token)+ '\n')
    if "'Masc'" in str(values) and "'Fem'" not in str(values):
        token=str(values).split(':')[1]
        if "'Sing'" in str(values):
         testset.write("T:o "+ str(token)+ '\n')
         testset.write("F:a "+ str(token)+ '\n')
        elif "'Plur'" in str(values):
         testset.write("T:os "+ str(token)+ '\n')
         testset.write("F:as "+ str(token)+ '\n')
