import sys
def splittrain_val(fileall,valratio=0.1):
    fileids = fileall.split('.')
    fileid = fileids[len(fileids)-2]
    f=open(fileall);
    ftrain=open(fileid+"_train.txt",'w');
    fval=open(fileid+"_val.txt",'w');
    count = 0
    if valratio == 0 or valratio >= 1:
        valratio = 0.1
    
    interval = (int)(1.0 / valratio)
    while 1:
        line = f.readline()
        if line:
            count = count + 1
            if count % interval == 0:
                fval.write(line)
            else:
                ftrain.write(line)
        else:
            break

splittrain_val(sys.argv[1],0.1)
