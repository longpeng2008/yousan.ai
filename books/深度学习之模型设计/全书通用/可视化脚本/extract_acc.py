import os
import sys
lines = open(sys.argv[1],'r').readlines()
fout = open(sys.argv[2],'w')
for line in lines:
   line = line.split('accuracy_at_1 = ')[-1]
   fout.write(line)

