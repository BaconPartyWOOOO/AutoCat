import random
import os.path
import numpy as np
import pickle

def savein(name, list):
    print("generating array: " + name)
    a = np.array([[0]*(5000 - len(x)) + x for x in list])
    print("saving")
    np.save(name, a)
    print("done")

def tokenize(text):
    tokens = []
    for i in range(len(text)):
        tokens.append(text[i:i + 1])
    return tokens

def process(vid, folder, d):
    data = []
    length = 0
    # filename = "F:\\Data\\Unsorted\\Comments\\" + format(folder, '04d') + "/" + vid + ".jsonl"
    filename = "/Users/masakura/Desktop/" + format(folder, '04d') + "/" + vid + ".jsonl"
    file = open(filename)
    with file as f:
        for line in f:
            line = line.replace(":\"", "").split("\"")
            for i in range(len(line)):
                if line[i] == "content":
                    comment = line[i + 1]
                    for t in tokenize(comment):
                        if t not in d:
                            d[t] = len(d) + 1
                            print(len(d), end="\r")
                        data.append(d[t])
                        length += 1
                        if length >= 6000000:
                            break
                    if length >= 6000000:
                        break
            if length >= 6000000:
                break
    return data

def preprocess():
    indata = []
    outdata = []
    intest = []
    outtest = []
    dc = 0
    ic = 0
    x = 0
    y = 0
    d = {}
    c = {}
    for folder in range(2957):
        # filename = "F:\\Data\\Unsorted\\Videos\\" + format(folder, '04d') + ".jsonl"
        filename = "/Users/masakura/Desktop/" + format(folder, '04d') + ".jsonl"
        if os.path.exists(filename):
            file = open(filename)
            with file as f:
                for line in f:
                    line = line.replace(":\"", "").split("\"")
                    for i in range(len(line)):
                        if line[i] == "video_id":
                            vid = line[i + 1]
                        if line[i] == "category":
                            cat = line[i + 1]
                            if cat != ":null,":
                                print(vid)
                                if os.path.exists("/Users/masakura/Desktop/" + format(folder, '04d') + "/" + vid + ".jsonl"):
                                # if os.path.exists("F:\\Data\\Unsorted\\Comments\\" + format(folder, '04d') + "/" + vid + ".jsonl"):
                                    if random.random() < .7:
                                        indata.append(process(vid, folder, d))
                                        if cat not in c:
                                            c[cat] = len(c) + 1
                                        outdata.append(c[cat])
                                        x += 1
                                        if x >= 10000:
                                            idname = 'indata' + str(dc) + '.npy'
                                            odname = 'outdata' + str(dc) + '.npy'
                                            savein(idname, np.array(indata))
                                            np.save(odname, np.array(outdata))
                                            indata = []
                                            outdata = []
                                            x = 0
                                            dc += 1
                                    else:
                                        intest.append(process(vid, folder, d))
                                        outtest.append(cat)
                                        y += 1
                                        if y >= 10000:
                                            itname = 'intest' + str(ic) + '.npy'
                                            otname = 'outtest' + str(ic) + '.npy'
                                            savein(itname, np.array(intest))
                                            np.save(otname, np.array(outtest))
                                            intest = []
                                            outtest = []
                                            y = 0
                                            ic += 1
                                else:
                                    print("comments not found")
                            break
        else:
            print("metadata not found")
    idname = 'indata' + str(dc) + '.npy'
    odname = 'outdata' + str(dc) + '.npy'
    savein(idname, np.array(indata))
    np.save(odname, np.array(outdata))
    itname = 'intest' + str(ic) + '.npy'
    otname = 'outtest' + str(ic) + '.npy'
    savein(itname, np.array(intest))
    np.save(otname, np.array(outtest))
    with open('chardict.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('catdict.pickle', 'wb') as handle:
        pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    preprocess()