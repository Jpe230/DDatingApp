import json
import json
from os import listdir
from os.path import isfile, join

files = ['./labels/' + f for f in listdir('./labels/') if isfile(join('./labels', f))]

with open('User_labels.txt', 'a') as userFile:
    for f in files:
        with open(f) as json_file:
            newData = json.load(json_file)
            
            normData = []

            # Normalize data to be between 0-5
            for d in newData:
                # Remove prefix
                path = d[0].replace("./Images/", "")
                score = float(d[1])/3

                result = path + " " + str(score) + "\n"

                #Write to file
                userFile.write(result)
                

