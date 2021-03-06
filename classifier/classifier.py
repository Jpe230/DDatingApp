import random
import json
from os import listdir
from os.path import isfile, join
import numpy

files = ['./Images/' + f for f in listdir('./Images') if isfile(join('./Images', f))]

random.shuffle(files)

l = numpy.array_split(numpy.array(files), 4)

idx = 0
for i in l:
    idx += 1
    filename = 'label{}.py'.format(idx)
    with open(filename, 'w') as outfile:
        json.dump(i.tolist(), outfile)