import numpy as np
import cv2

#imgFlag = cv2.IMREAD_COLOR
imgFlag = cv2.IMREAD_GRAYSCALE
img = cv2.imread("LOTF_GS_RECTANGLED.jpg", imgFlag)
maskFlag = cv2.IMREAD_GRAYSCALE
mask = cv2.imread("LOTF_GS_RECTANGLES.jpg", maskFlag)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()