import cv2
import numpy as np

id = 0

I = cv2.imread('Huang and Liu/'+str(id).zfill(5)+'_Input.png').astype(np.float32)/255.0
B = cv2.imread('Huang and Liu/'+str(id).zfill(5)+'_Background_Liu2020.png').astype(np.float32)/255.0

# A = (np.abs(I - B) > 0.2).astype(np.float32)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

A = (np.abs(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)))
A = np.clip(A * 10, 0.0, 1.0)
A = A[:, :, None]
A = cv2.morphologyEx(A, cv2.MORPH_OPEN, kernel)
A = A[:, :, None]

# cv2.imwrite('Huang and Liu/'+str(id).zfill(5)+'_A.png', np.round(A*255.0).astype(np.uint8))

gray = np.ones_like(B) * 0.5

F = I * A + gray * (1.0 - A)

cv2.imwrite('Huang and Liu/'+str(id).zfill(5)+'_Occlusion_Liu2020.png', np.round(F*255.0).astype(np.uint8))

