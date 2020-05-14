import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import seaborn as sns
import argparse
from imutils import paths
import cv2

     
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of images")
args = vars(ap.parse_args())

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(8,2)
path1 = args["dataset"]
listing = os.listdir(path1)
for x,i in enumerate(listing):
    data = []
    # imagePaths = sorted(list(paths.list_images(str(args["dataset"]) + '\\' + str(i))))
    imagePaths = sorted(os.listdir(str(args["dataset"]) + '\\' + str(i)))
    for imagePath in imagePaths:
        # load the image, resize the image to be 64x64 pixels (ignoring
        # aspect ratio), flatten the image
        # into a list, and store the image in the data list
        image = cv2.imread(str(args["dataset"]) + '\\' + str(i) + '\\' + imagePath, 0)
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image = cv2.resize(image, (64, 64))
        data.append(image.flatten())
    np_img = np.array(np.transpose(data))
    np_img_scaled= (np_img-np_img.mean())/np_img.std()
    U, s, V = np.linalg.svd(np_img_scaled)
    V = np.asmatrix(V)
    rep = np.transpose(data)*V[:,0]
    mat = np.reshape(rep,(64,64))
    # img = Image.fromarray(np.uint8(mat * 255) , 'L')
    axarr[round(x/2)][x%2].imshow(mat)
    # axarr[x] = img.show()
plt.show()