import numpy as np
train_file = "trainingdataset.txt"

#open file
file1 = open(train_file, 'r')
lines = file1.readlines()

#split lines to formal and informal
indexFormal = [i for i in range(0,407,3)]
indexCasual = [j for j in range(1,408,3)]

arrForInd = np.array(lines)
arrCasInd = np.array(lines)

formalLines = arrForInd[indexFormal]
casualLines = arrCasInd[indexCasual]

#test output for formal and informal
print(len(formalLines))
print(len(casualLines))
