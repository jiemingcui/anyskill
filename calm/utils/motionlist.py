import json
import numpy as np
import pandas as pd
import os


def extract_babel():
    babelMotion = []
    with open("/home/cjm/CALM/motionlist/val.json", "r") as f:
        babel = json.load(f)

    for data in list(babel.keys()):
        if babel[data]["frame_ann"] is not None:
            motions = babel[data]["frame_ann"]["labels"]
            if len(motions) != 1:
                for motion in motions:
                    babelMotion.append(motion["raw_label"])
        # else:
        #     babelMotion.append(motions["raw_label"])
    babelMotion = list(set(babelMotion))
    with open("./motionlist/babel.txt", "w") as a:
        for i in babelMotion:
            a.write(i + "\n")

def extract_kit(path):
    kitMotion = []
    for i in os.listdir(path):
        f = open(path+i, "r")
        text = f.readlines()
        # if isinstance(text,list):
        if (type(text).__name__ == 'list'):
            for t in text:
                kitMotion.append(str(t).split("#")[0])
        else:
            kitMotion.append(text.split("#")[0])

    with open("./motionlist/kit.txt", "w") as a:
        for i in kitMotion:
            a.write(i + "\n")

    print(0)

def extract_ml(path):
    mlMotion = []
    for i in os.listdir(path):
        f = open(path+i, "r")
        text = f.readlines()
        # if isinstance(text,list):
        if (type(text).__name__ == 'list'):
            for t in text:
                mlMotion.append(str(t).split("#")[0])
        else:
            mlMotion.append(text.split("#")[0])

    with open("./motionlist/ml.txt", "w") as a:
        for i in mlMotion:
            a.write(i + "\n")

    print(0)

if __name__ == "__main__":
    # extract_kit("/home/cjm/CALM/motionlist/texts_kit/")
    extract_ml("/home/cjm/CALM/motionlist/texts_ml/")

# print(0)


