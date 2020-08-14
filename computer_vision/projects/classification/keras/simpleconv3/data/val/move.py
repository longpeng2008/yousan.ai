import os
import sys
import shutil
lines = open(sys.argv[1],'r')
for line in lines:
    line = line.strip()
    image,label = line.split(' ')
    id = image.split('/')[-1]
    shutil.copyfile(image,os.path.join(label,id))
