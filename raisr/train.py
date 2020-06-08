import cv2
import numpy as np
import os
import pickle
import sys
from cgls import cgls
from filterplot import filterplot
from gaussian2d import gaussian2d
from gettrainargs import gettrainargs
from hashkey import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform

import multiprocessing

args = gettrainargs()

# Define parameters
R = int(args.scaling)
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = '../data/train'

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def processQV(image):
    Q = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize))
    V = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))

    print('\r', end='')
    print(' ' * 60, end='')
    print('\rProcessing new image')
    origin = cv2.imread(image)
    # Extract only the luminance in YCbCr
    grayorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)[:,:,0]
    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    # Downscale (bicubic interpolation)
    height, width = grayorigin.shape
    LR = transform.resize(grayorigin, (floor((height+1)/2),floor((width+1)/2)), mode='reflect', anti_aliasing=False)
    # Upscale (bilinear interpolation)
    height, width = LR.shape
    heightgrid = np.linspace(0, height-1, height)
    widthgrid = np.linspace(0, width-1, width)
    bilinearinterp = interpolate.interp2d(widthgrid, heightgrid, LR, kind='linear')
    heightgrid = np.linspace(0, height-1, height*2-1)
    widthgrid = np.linspace(0, width-1, width*2-1)
    upscaledLR = bilinearinterp(widthgrid, heightgrid)
    # Calculate A'A, A'b and push them into Q, V
    height, width = upscaledLR.shape
    operationcount = 0
    totaloperations = (height-2*margin) * (width-2*margin)
    for row in range(margin, height-margin):
        for col in range(margin, width-margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                sys.stdout.flush()
            operationcount += 1
            # Get patch
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            patch = np.matrix(patch.ravel())
            # Get gradient block
            gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            gy, gx = np.gradient(gradientblock)
            angle, strength, coherence = hashkey(gy, gx, Qangle, weighting)
            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
            # Get corresponding HR pixel
            pixelHR = grayorigin[row,col]
            # Compute A'A and A'b
            ATA = np.dot(patch.T, patch)
            ATb = np.dot(patch.T, pixelHR)
            ATb = np.array(ATb).ravel()
            # Compute Q and V
            Q[angle,strength,coherence,pixeltype] += ATA
            V[angle,strength,coherence,pixeltype] += ATb
    
    return Q, V

def run():
    Q = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize))
    V = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
    h = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))

    donelist = None

    # Read Q,V from file
    if args.qmatrix:
        with open(args.qmatrix, "rb") as fp:
            Q = pickle.load(fp)
    if args.vmatrix:
        with open(args.vmatrix, "rb") as fp:
            V = pickle.load(fp)

    # Get image list
    imagelist = []
    for parent, dirnames, filenames in os.walk(trainpath):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))

    imagecount = 1

    if args.done_file:
        with open(args.done_file, "r") as f:
            donelist = f.read().splitlines()
        imagelist = [x for x in imagelist if x not in donelist]
        imagecount += len(donelist)

    pool = multiprocessing.Pool()
    chunk_size = max(multiprocessing.cpu_count() * 2, 20)
    for img_chunk in chunks(imagelist, chunk_size):
        print('Processing images {}-{} of {}'.format(imagecount, min(imagecount+chunk_size, len(imagelist)), len(imagelist)))
        for imgpathname in img_chunk:
            print(imgpathname)

        qvlist = pool.map(processQV, img_chunk)
        for q, v in qvlist:
            Q = np.add(Q, q)
            V = np.add(V, v)

        print('\nWriting Q and V mid-train')
        with open("filters/q{}.p".format(R), "w+b") as fp:
            pickle.dump(Q, fp)
        with open("filters/v{}.p".format(R), "w+b") as fp:
            pickle.dump(V, fp)

        imagecount += chunk_size

    pool.close()
    pool.join()

    # Write Q,V to file
    with open("filters/q{}.p".format(R), "w+b") as fp:
        pickle.dump(Q, fp)
    with open("filters/v{}.p".format(R), "w+b") as fp:
        pickle.dump(V, fp)

    # Preprocessing permutation matrices P for nearly-free 8x more learning examples
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rPreprocessing permutation matrices P for nearly-free 8x more learning examples ...')
    sys.stdout.flush()
    P = np.zeros((patchsize*patchsize, patchsize*patchsize, 7))
    rotate = np.zeros((patchsize*patchsize, patchsize*patchsize))
    flip = np.zeros((patchsize*patchsize, patchsize*patchsize))
    for i in range(0, patchsize*patchsize):
        i1 = i % patchsize
        i2 = floor(i / patchsize)
        j = patchsize * patchsize - patchsize + i2 - patchsize * i1
        rotate[j,i] = 1
        k = patchsize * (i2 + 1) - i1 - 1
        flip[k,i] = 1
    for i in range(1, 8):
        i1 = i % 4
        i2 = floor(i / 4)
        P[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))
    Qextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize))
    Vextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
    for pixeltype in range(0, R*R):
        for angle in range(0, Qangle):
            for strength in range(0, Qstrength):
                for coherence in range(0, Qcoherence):
                    for m in range(1, 8):
                        m1 = m % 4
                        m2 = floor(m / 4)
                        newangleslot = angle
                        if m2 == 1:
                            newangleslot = Qangle-angle-1
                        newangleslot = int(newangleslot-Qangle/2*m1)
                        while newangleslot < 0:
                            newangleslot += Qangle
                        newQ = P[:,:,m-1].T.dot(Q[angle,strength,coherence,pixeltype]).dot(P[:,:,m-1])
                        newV = P[:,:,m-1].T.dot(V[angle,strength,coherence,pixeltype])
                        Qextended[newangleslot,strength,coherence,pixeltype] += newQ
                        Vextended[newangleslot,strength,coherence,pixeltype] += newV
    Q += Qextended
    V += Vextended

    # Compute filter h
    print('Computing h ...')
    sys.stdout.flush()
    operationcount = 0
    totaloperations = R * R * Qangle * Qstrength * Qcoherence
    for pixeltype in range(0, R*R):
        for angle in range(0, Qangle):
            for strength in range(0, Qstrength):
                for coherence in range(0, Qcoherence):
                    if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                        print('\r|', end='')
                        print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                        print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                        print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                        sys.stdout.flush()
                    operationcount += 1
                    h[angle,strength,coherence,pixeltype] = cgls(Q[angle,strength,coherence,pixeltype], V[angle,strength,coherence,pixeltype])

    # Write filter to file
    with open("filters/filter{}.p".format(R), "wb") as fp:
        pickle.dump(h, fp)

    # Plot the learned filters
    if args.plot:
        filterplot(h, R, Qangle, Qstrength, Qcoherence, patchsize)

    print('\r', end='')
    print(' ' * 60, end='')
    print('\rFinished.')

if __name__ == '__main__':
    run()
