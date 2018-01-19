from __future__ import print_function
import cv2
from PIL import Image
from time import time
import os, sys
import numpy as np
import pillowfight
import pickle
import string
import re
import argparse
import tempfile
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
args = vars(ap.parse_args())

print("**Loading and preprocessig**")
image = cv2.imread(args["image"])
augs = list()

h, w, c = image.shape
asp = w/h
if(w>h):
    w = 650
    h = w/asp
else:
    h = 650
    w = asp*h
    
image = cv2.resize(image, (int(w), int(h)))
print("Image resized to %s X %s" % (int(w), int(h)))

kernel = np.ones((5,5), np.uint8)

#gaussian blur
gaus = cv2.GaussianBlur(image, (5,5), 0)
gaus = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#otsu without inversion
(thresh, ots) = cv2.threshold(gaus, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
eroded = cv2.erode(ots, kernel, iterations = 1)
augs.extend([ots, eroded])

#otsu with updated threshold
(pos_thresh, pos_ots) = cv2.threshold(gaus, thresh + 60, 255, cv2.THRESH_BINARY)
pos_eroded = cv2.erode(pos_ots, kernel, iterations = 1)
augs.extend([pos_ots, pos_eroded])

#otsu with updated threshold
(neg_thresh, neg_ots) = cv2.threshold(gaus, thresh - 60, 255, cv2.THRESH_BINARY)
neg_eroded = cv2.erode(neg_ots, kernel, iterations = 1)
augs.extend([neg_ots, neg_eroded])

#otsu with inversion
(inv_thresh, inv_ots) = cv2.threshold(gaus, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
inv_eroded = cv2.erode(inv_ots, kernel, iterations = 1)
augs.extend([inv_ots, inv_eroded])

#inverted otsu with updated threshold
(inv_pos_thresh, inv_pos_ots) = cv2.threshold(gaus, inv_thresh + 60, 255, cv2.THRESH_BINARY_INV)
inv_pos_eroded = cv2.erode(inv_pos_ots, kernel, iterations = 1)
augs.extend([inv_pos_ots, inv_pos_eroded])

#inverted otsu with updated threshold
(inv_neg_thresh, inv_neg_ots) = cv2.threshold(gaus, inv_thresh - 60, 255, cv2.THRESH_BINARY_INV)
inv_neg_eroded = cv2.erode(inv_neg_ots, kernel, iterations = 1)
augs.extend([inv_neg_ots, inv_neg_eroded])

tempdir = tempfile.mkdtemp()
temptxt = tempfile.NamedTemporaryFile(mode='w', delete=True)

with open(temptxt.name, 'w') as fp:
    for i, img in enumerate(augs):
        pil_image = Image.fromarray(img)
        swt_image = pillowfight.swt(pil_image, output_type = pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)
        filename = str(tempdir + "/" + str(i+1) + '.png')
        swt_image.save(filename, format = 'png')
        fp.write((filename+"\n"))
    
os.system('tesseract '+ temptxt.name + ' ~/optt.txt --oem 2')

temptxt.close()
shutil.rmtree(tempdir)


