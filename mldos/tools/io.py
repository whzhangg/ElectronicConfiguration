import os
import json


def change_filetype(name, content):
    parts=name.split(".")
    newname=""
    part_len=len(parts)
    for j,part in enumerate(parts):
        if j < part_len-1:
            newname+=part
            newname+="."
        else:
            newname+=content
    return newname


def write_json(dictionary, filename):
    f=open(filename,'w')
    json.dump(dictionary,f,indent="    ")
    f.close()


def read_json(filename):
    f=open(filename,'r')
    data=json.load(f)
    f.close()
    return data


def printProgressBar(iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def make_target_directory(directory):
    to_create = []
    last_existing = directory
    while not os.path.exists(last_existing):
        parts = os.path.split(last_existing)
        last_existing = parts[0]
        to_create.append(parts[1])
        if last_existing == "":
            break
    if to_create:
        path_to_create = last_existing
        for i in range(len(to_create)):
            path_to_create = os.path.join(path_to_create, to_create[-i-1])
            os.makedirs(path_to_create)
    return


