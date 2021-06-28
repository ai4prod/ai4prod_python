

from skimage import io, transform, color

import cv2

from matplotlib import pyplot as plt
import numpy as np



a = np.array([10, 11, 13, 15])
b = np.array([10,  9,  9, 11])

common = a[np.isclose(a,b,atol=2,rtol=0)]

output_size=320
image= cv2.imread("onnx/test.jpg")
image_cv= image.copy()


image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)



h, w = image.shape[:2]

if isinstance(output_size,int):
    if h > w:
        new_h, new_w = output_size*h/w,output_size
    else:
        new_h, new_w = output_size,output_size*w/h
else:
    new_h, new_w = output_size

new_h, new_w = int(new_h), int(new_w)


output_shape= (320,320,3)

output_shape = tuple(output_shape)
input_shape = image.shape

factors = (np.asarray(input_shape, dtype=float) /
               np.asarray(output_shape, dtype=float))

truncate= 4.0
sigma = np.maximum(0, (factors - 1) / 2)

print(sigma)
print(image.shape)

input("sigma")
sizeB= int(truncate * sigma[2] + 0.5)

if(sizeB==0):
    sizeB=sizeB+1


sizeG= int(truncate * sigma[1] + 0.5)

if (sizeG % 2) == 0:  
  sizeG=sizeG+1
else:  
   print("{0} is Odd number".format(sizeG))  
sizeR= int(truncate * sigma[0] + 0.5)

if (sizeR % 2) == 0:  
  sizeR=sizeR+1
else:  
   print("{0} is Odd number".format(sizeR))  


ksizeB= (sizeB,1)
ksizeG= (sizeG,1)
ksizeR= (sizeR,1)

print(ksizeB)
print(ksizeR)



(B, G, R) = cv2.split(image_cv)


# gaussian 2d kernel
# B = cv2.GaussianBlur(B, ksizeB, sigma[1], borderType=cv2.BORDER_CONSTANT)
# G = cv2.GaussianBlur(G, ksizeG, sigma[1], borderType=cv2.BORDER_CONSTANT)
# R = cv2.GaussianBlur(R, ksizeR, sigma[2], borderType=cv2.BORDER_CONSTANT)

#gaussian 1dkernel

# kernelB= cv2.getGaussianKernel(ksizeB[0],sigma[2])
# kernelG= cv2.getGaussianKernel(ksizeG[0],sigma[1])
# kernelR= cv2.getGaussianKernel(ksizeR[0],sigma[0])

# B = cv2.filter2D(B, -1, kernelB)
# G = cv2.filter2D(G, -1, kernelG)
# R = cv2.filter2D(R, -1, kernelR)



image_cv = cv2.merge([B, G, R])

image_cv= cv2.resize(image_cv,(output_size,output_size),interpolation=cv2.INTER_NEAREST)
image_cv1= cv2.resize(image_cv,(output_size,output_size),interpolation=cv2.INTER_CUBIC)
image_cv2= cv2.resize(image_cv,(output_size,output_size),interpolation=cv2.INTER_LINEAR)
image_cv3= cv2.resize(image_cv,(output_size,output_size),fx=0.5,fy=0.5,interpolation=cv2.INTER_LANCZOS4) 
image_cv4= cv2.resize(image_cv,(output_size,output_size),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA) 

img = transform.resize(image,(output_size,output_size),mode='constant')

#img= img*255

#img= img.astype(int)

img= cv2.cvtColor(img.astype('float32'),cv2.COLOR_RGB2BGR)

print(img.shape)


cv2.imshow("scikit",img)
cv2.imshow("opencv_NEAREST",image_cv)
#cv2.imshow("opencv1_CUBIC",image_cv1)
cv2.imshow("opencv2_LINEAR",image_cv2)
cv2.imshow("opencv3_LANCZOS",image_cv3)
#cv2.imshow("opencv3_AREA",image_cv4)
cv2.imwrite("opencvAREA.jpg",image_cv4)
cv2.waitKey(0)
#print(img[np.allclose(image_cv,img,atol=2)])



# cv2.imshow("image",image_cv)

# cv2.waitKey(1)

# io.imshow(img)
# plt.show()
