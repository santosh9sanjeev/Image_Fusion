import pywt
import cv2
import numpy as np

# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):
    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []
    return cooef


# Params
FUSION_METHOD = 'mean' # Can be 'min' || 'max || anything you choose according theory

# Read the two images
image1 = input("Enter image 1 name : ")
image2 = input("Enter image 2 name : ")
fuse_image = input("Enter fuse image name : ")

I1 = cv2.imread(image1,1)
I2 = cv2.imread(image2,1)
# cv2.imshow("frame",I1)
# cv2.imshow("frame1",I2)
# cv2.imwrite('pic(1).png',I1)
# cv2.imwrite('pic(2).png',I2)

# We need to have both images the same size
#I2 = cv2.resize(I2,I1.shape) # This is done when images size is not same

## Fusion algorithm

channel1=I1[:,:,0]
channel2=I1[:,:,1]
channel3=I1[:,:,2]
final=np.dstack((channel1,channel2,channel3))
# cv2.imshow("framefinal",final)

# First: Do wavelet transform on each image
wavelet = 'db1'
cooef1 = pywt.wavedec2(I1[:,:,0], wavelet)
cooef2 = pywt.wavedec2(I2[:,:,0], wavelet)

# Second: for each level in both image do the fusion according to the desire option
fusedCooef = []
for i in range(len(cooef1)-1):

    # The first values in each decomposition is the approximation values of the top level
    if(i == 0):
        fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))
    else:
        # For the rest of the levels we have tuples with 3 coefficients
        c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],FUSION_METHOD)
        c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
        c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)
        fusedCooef.append((c1,c2,c3))

# Third: After we fused the coefficient we need to transfer back to get the image
fusedImage = pywt.waverec2(fusedCooef, wavelet)

# Forth: normalize values to be in uint8
fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
fusedImage = fusedImage.astype(np.uint8)

# Fifth: Show image
# cv2.imshow("win",fusedImage)
# cv2.waitKey(0)
# cv2.imwrite('fusedimg1.png',fusedImage)



wavelet = 'db1'
cooef3 = pywt.wavedec2(I1[:,:,1], wavelet)
cooef4 = pywt.wavedec2(I2[:,:,1], wavelet)

# Second: for each level in both image do the fusion according to the desire option
fusedCooef01 = []
for i in range(len(cooef3)-1):

    # The first values in each decomposition is the approximation values of the top level
    if(i == 0):
        fusedCooef01.append(fuseCoeff(cooef3[0],cooef4[0],FUSION_METHOD))
    else:
        # For the rest of the levels we have tuples with 3 coefficients
        c11 = fuseCoeff(cooef3[i][0],cooef4[i][0],FUSION_METHOD)
        c21 = fuseCoeff(cooef3[i][1], cooef4[i][1], FUSION_METHOD)
        c31 = fuseCoeff(cooef3[i][2], cooef4[i][2], FUSION_METHOD)

        fusedCooef01.append((c11,c21,c31))

# Third: After we fused the coefficient we need to transfer back to get the image
fusedImage1 = pywt.waverec2(fusedCooef01, wavelet)

# Forth: normmalize values to be in uint8
fusedImage1 = np.multiply(np.divide(fusedImage1 - np.min(fusedImage1),(np.max(fusedImage1) - np.min(fusedImage1))),255)
fusedImage1 = fusedImage1.astype(np.uint8)

# Fifth: Show image
# cv2.imshow("win1",fusedImage1)
# cv2.waitKey(0)
# cv2.imwrite('fusedimg2.png',fusedImage1)


wavelet = 'db1'
cooef5 = pywt.wavedec2(I1[:,:,2], wavelet)
cooef6 = pywt.wavedec2(I2[:,:,2], wavelet)

# Second: for each level in both image do the fusion according to the desire option
fusedCooef02 = []
for i in range(len(cooef5)-1):

    # The first values in each decomposition is the approximation values of the top level
    if(i == 0):
        fusedCooef02.append(fuseCoeff(cooef5[0],cooef6[0],FUSION_METHOD))

    else:

        # For the rest of the levels we have tuples with 3 coefficients
        c12 = fuseCoeff(cooef5[i][0],cooef6[i][0],FUSION_METHOD)
        c22 = fuseCoeff(cooef5[i][1], cooef6[i][1], FUSION_METHOD)
        c32 = fuseCoeff(cooef5[i][2], cooef6[i][2], FUSION_METHOD)

        fusedCooef02.append((c12,c22,c32))

# Third: After we fused the coefficient we need to transfer back to get the image
fusedImage2 = pywt.waverec2(fusedCooef02, wavelet)

# Forth: normalize values to be in uint8
fusedImage2 = np.multiply(np.divide(fusedImage2 - np.min(fusedImage2),(np.max(fusedImage2) - np.min(fusedImage1))),255)
fusedImage2 = fusedImage2.astype(np.uint8)

# Fifth: Show image
# cv2.imshow("win2",fusedImage2)
# cv2.waitKey(0)
# cv2.imwrite('fusedimg3.png',fusedImage2)

channel11=fusedImage[:,:]
channel12=fusedImage1[:,:]
channel13=fusedImage2[:,:]
final1=np.dstack((channel11,channel12,channel13))
# cv2.imshow("framefinale",final1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(fuse_image, final1)
