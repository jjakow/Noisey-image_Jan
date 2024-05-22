'''
Run this script against the folder from the experiment
at ./src/data/tmp/runs/...
example: python makeBetterGraph.py --folder src/data/tmp/runs/exp_28/
'''

import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import yaml

def makemAPGraph(path, augName, modelName, isCompound): #isCompound:bool
    path = "./" + path
    map50s = np.load( os.path.join(path, 'graphing.npy'), allow_pickle=True )
    map50s = map50s.item()
    with open( os.path.join(path, 'meta.yaml') , 'r') as f:
        metadata = yaml.safe_load(f)
    plt.figure(figsize=(7.5,5))

    # Compounding only - xaxis determined by fixed number set (change later - needs to scale)
    if isCompound:
        xaxis = [1, 2, 3, 4, 5]
        mAP = map50s['Compound'][0]
        plt.plot(xaxis, mAP, '-o')
    # Standard graph preparation
    else:
        for i, noise in enumerate(metadata.keys()): # Use isCompound to ignore this process
            xaxis = metadata[noise]
            mAP = map50s[noise][0]
            plt.plot(xaxis, mAP,'-o')

    plt.title(augName + ' vs ' + modelName + ' mAP')
    plt.ylabel('mAP')
    plt.xlabel('Augment Level')
    plt.savefig(path + "/betterGraph.jpg")
    return

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--folder', '-f', type=str,)
#     args = parser.parse_args()

#     map50s = np.load( os.path.join(args.folder, 'graphing.npy') )
#     print(map50s)
#     with open( os.path.join(args.folder, 'meta.yaml') , 'r') as f:
#         metadata = yaml.safe_load(f)

#     plt.figure(figsize=(7.5,5))
#     for i, noise in enumerate(metadata.keys()):
#         xaxis = metadata[noise]
#         mAP = map50s[i]
#         plt.plot(xaxis, mAP,'-o')

#     plt.title('Gaussian Noise vs YOLOv3 mAP')
#     plt.ylabel('mAP')
#     plt.xlabel('Standard Deviation')
#     plt.savefig(args.folder + "betterGraph.jpg")
    