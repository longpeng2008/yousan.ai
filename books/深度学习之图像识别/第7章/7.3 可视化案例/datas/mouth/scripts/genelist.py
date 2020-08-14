# _*_ coding:utf8 _*_
import os
import sys
def listfiles(rootDir,txtfile,label=0):
    ftxtfile = open(txtfile,'w')
    list_dirs = os.walk(rootDir)
    count = 0
    dircount = 0
    for root, dirs, files in list_dirs:
        for d in dirs:
            print os.path.join(root,d)
            dircount = dircount + 1
        for f in files:
            print os.path.join(root,f)
            ftxtfile.write(os.path.join(root,f)+' '+str(label)+'\n')
            count = count + 1
    print rootDir+"has"+str(count)+"files"

listfiles(sys.argv[1],sys.argv[2],sys.argv[3])

