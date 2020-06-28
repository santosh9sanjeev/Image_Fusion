import cv2
import math
import numpy as np


def calculate_metrics(resized1, resized2, Ifinal):
    #subtract here
    I31=cv2.subtract(Ifinal,resized1)
    I32=cv2.subtract(Ifinal,resized2)

    #show subtacted images
    cv2.imshow("framemetric1",I31)
    cv2.imshow("framemetric2",I32)


    a1=0
    for i in range(0,480):
        for j in range(0,640):
            a1=a1 + ((I31[i][j])**2)
    # print(a1)
    mean_square1=a1/(640*480)
    print("Mean Square 1 : ", mean_square1)

    PSNR1=10*(math.log10(255**2/mean_square1))
    print("PSNR 1 : ", PSNR1)


    a2 = 0
    for i in range(0,480):
        for j in range(0,640):
            a2=a2 + ((I32[i][j])**2)
    # print(a2)
    mean_square2=a2/(640*480)
    print("Mean Square 2 : ",mean_square2)

    PSNR2=10*(math.log10(255**2/mean_square2))
    print("PSNR 2 : ",PSNR2)


    #averaged MSE and PSNR

    MSE=(mean_square1+mean_square2)/2
    PSNR=(PSNR1+PSNR2)/2

    #   CC
    lisx1 = []
    lisx2 = []
    lisy = []
    for i in range(0, 480):
        for j in range(0, 640):
            lisx1.append(resized1[i][j])
            lisx2.append(resized2[i][j])
            lisy.append(Ifinal[i][j])
    kinp1 = 0
    kinp2 = 0
    kout = 0
    # print(len(lisx1))
    for i in range(len(lisx1)):
        kinp1 = kinp1 + lisx1[i]
        kinp2 = kinp2 + lisx2[i]
        kout = kout + lisy[i]
    avgx1 = kinp1 // len(lisx1)
    avgx2 = kinp2 // len(lisx2)
    avgy = kout // len(lisy)

    CCx1y1 = 0
    CCx1 = 0
    CCy1 = 0
    for i in range(0, 480):
        for j in range(0, 640):
            CCx1y1 = CCx1y1 + ((resized1[i][j] - avgx1) * (Ifinal[i][j] - avgy))

            CCx1 = CCx1 + ((resized1[i][j] - avgx1) ** 2)
            CCy1 = CCy1 + ((Ifinal[i][j] - avgy) ** 2)

    CC1 = (CCx1y1 / ((math.sqrt(CCx1)) * (math.sqrt(CCy1))))
    print("Correlation Coefficient 1 : ",CC1)

    CCx2y2 = 0
    CCx2 = 0
    CCy2 = 0
    for i in range(0, 480):
        for j in range(0, 640):
            CCx2y2 = CCx2y2 + ((resized2[i][j] - avgx2) * (Ifinal[i][j] - avgy))

            CCx2 = CCx2 + ((resized2[i][j] - avgx2) ** 2)
            CCy2 = CCy2 + ((Ifinal[i][j] - avgy) ** 2)

    CC2 = (CCx2y2 / ((math.sqrt(CCx2)) * (math.sqrt(CCy2))))
    print("Correlation Coefficient 2 : ",CC2)

    # averaged CC
    CC = (CC1 + CC2) / 2

    print("MSE  : ", MSE)
    print("PSNR : ", PSNR)
    print("Correlation Coefficient: ",CC)


#give input images of cat and dog in black and white
image1 = input("Enter image 1 name : ")
image2 = input("Enter image 2 name : ")
I1 = cv2.imread(image1,0)
I2 = cv2.imread(image2,0)

#resize all images to same dimensions then only subtract work

resized1=cv2.resize(I1,(640,480),interpolation=cv2.INTER_AREA)
resized2=cv2.resize(I2,(640,480),interpolation=cv2.INTER_AREA)

#input images after resizing
cv2.imshow("resized1",resized1)
cv2.imshow("resized2",resized2)

#give ur output image here 3 cases vi separately in black and white
n = int(input("Enter n value : "))
for i in range(n):
    fuse_image = input("Enter fuse image name : ")
    Ifinal=cv2.imread(fuse_image,0)
    Ifinal=cv2.resize(Ifinal,(640,480),interpolation=cv2.INTER_AREA)

    cv2.imshow("final_Bw",Ifinal)
    calculate_metrics(resized1, resized2, Ifinal)
