import shutil
import os
from pathlib import Path

def addFnametoList(path):
    fullList = [x for x in os.listdir(path)]
    count = len(fullList)//20 #subset of list of the full list
    tmplist = []
    
    for fn in fullList:
        if len(tmplist) < count:
            prefix = fn[:-8]
            newList = [x for x in os.listdir(path) if x[:-8] == prefix] #grab all 20 views
#             print(prefix, len(newList))
            tmplist += newList
    print('Grabbing ', len(tmplist), ' files from ', path)
    return tmplist

path = Path('./ModelNet40_20')
#get all the subfolders
folders = [f.path for f in os.scandir(path) if f.is_dir()]
folders = folders[0:3]
subfolders = []
for folds in folders:
    subfolders += [f.path for f in os.scandir(folds) if f.is_dir()]

#for each subfolder copy over some of the files. Number determined by count = len(fullList)//x line.
for subs in subfolders:
    sourcePath = Path(subs)
    destinationPath = Path(subs[0:13]+'Small/'+subs[14:])
    if not os.path.exists(destinationPath):
        os.makedirs(destinationPath)
    fnlist = addFnametoList(sourcePath)
    for fn in fnlist:
        shutil.copy2(sourcePath/fn, destinationPath)