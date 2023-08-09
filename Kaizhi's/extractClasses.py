'''
Created on May 30, 2023

@author: Haren L
'''

classesFile = open('originalClasses')
writeFile = open('classes.txt', 'w')
line = classesFile.readline()
line = classesFile.readline()
line = classesFile.readline()
while line:
    line = line.split(':')
    print(line)
    writeFile.write(line[1][1:])
    line = classesFile.readline()
classesFile.close()
writeFile.close()
if __name__ == '__main__':
    pass