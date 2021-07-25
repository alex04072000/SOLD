import numpy as np
import cv2

L = cv2.imread(r'C:\Users\user\Documents\Alex\Research\VideoReflectionRemoval_revisit\Obstruction_HTML_CameraReady_PAMI\results\raindrop\Huang and Liu\00002_Input.png')
R = cv2.imread(r'C:\Users\user\Documents\Alex\Research\VideoReflectionRemoval_revisit\Obstruction_HTML_CameraReady_PAMI\results\raindrop\Huang and Liu\00002_Background_Ours.png')

width = L.shape[1]
line_width = round(float(15)*float(L.shape[1]) / float(1152))


mask = np.zeros(L.shape, dtype=np.uint8)
roi_corners = np.array([[(0,0), (width//3-line_width//2,0), (2*width//3 - line_width//2, L.shape[0]), (0,L.shape[0])]], dtype=np.int32)
# roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
# fill the ROI so it doesn't get wiped out when the mask is applied
channel_count = L.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask, roi_corners, ignore_mask_color)

mask2 = np.zeros(L.shape, dtype=np.uint8)
roi_corners = np.array([[(width//3+line_width//2,0), (width,0), (L.shape[1], L.shape[0]), (2*width//3 + line_width//2,L.shape[0])]], dtype=np.int32)
# roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
# fill the ROI so it doesn't get wiped out when the mask is applied
channel_count = L.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask2, roi_corners, ignore_mask_color)

mask3 = np.zeros(L.shape, dtype=np.uint8)
roi_corners = np.array([[(width//3-line_width//2,0), (width//3+line_width//2,0), (2*width//3 + line_width//2,L.shape[0]), (2*width//3 - line_width//2, L.shape[0])]], dtype=np.int32)
# roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
# fill the ROI so it doesn't get wiped out when the mask is applied
channel_count = L.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask3, roi_corners, ignore_mask_color)

# masked_image = cv2.bitwise_and(L, mask)

print(np.max(mask))

O = L*(mask/255) + R * (mask2/255) + np.ones_like(R, np.uint8)*(mask3)

cv2.imwrite(r'raindrop.png', O)
