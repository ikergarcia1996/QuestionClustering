import pandas as pd
from sklearn.model_selection import train_test_split


quora = pd.read_csv('train.csv', header=1, sep=',')

train, test = train_test_split(quora, train_size=0.2)
train, test = train_test_split(train, train_size=0.8)
print('Train len: ' + str(len(train)))
print('Test len: ' + str(len(test)))

with open('train_set.csv','w+') as file:
    for index, line in train.iterrows():
        try:
            _,_,_,a,b,g = line
            a.replace('\t', ' ')
            b.replace('\t', ' ')

            print(a.rstrip() + '\t' + b.rstrip() + '\t' +str(g).rstrip(), file=file)
        except:
            print('Error in line ' + str(index) + ':' + str(line.values))

with open('dev_set.csv', 'w+') as file:
    for index, line in test.iterrows():
        try:
            _,_,_,a,b,g = line
            a.replace('\t', ' ')
            b.replace('\t', ' ')

            print(a.rstrip() + '\t' + b.rstrip() + '\t' +str(g).rstrip(), file=file)
        except:
            print('Error in line ' + str(index) + ':' + str(line.values))



