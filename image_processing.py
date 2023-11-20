import cv2
import numpy as np
import os

path_image = "./image_12mgpx_20231106-153147.jpg"
#path_image = "./image_12mgpx_20231117-1.jpg"

img = cv2.imread(path_image)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray_img,50,100)


# Trasformar a modelo de color hsv
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# rango bajo y alto hsv
lower_hsv = np.array([15, 0, 50])
upper_hsv = np.array([30, 255, 255])
# mascara 

mask_yellow = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
mask = mask_yellow
## rango bajo y alto hsv
#lower_hsv = np.array([0, 0, 50])
#upper_hsv = np.array([0, 255, 255])
#mask_yellow = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

#
contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Ordenamos por el area
page = sorted(contours, key=cv2.contourArea, reverse=True)

print(cv2.contourArea(page[0]))
print(cv2.contourArea(page[1]))

img_contours = cv2.drawContours(img.copy(), page, 0 , (0, 0, 255), 3)


#epsilon = 0.06 * cv2.arcLength(page[0], True)
for i in range(10):
    epsilon = 0.01 * i * cv2.arcLength(page[0], True)
    corners = cv2.approxPolyDP(page[0], epsilon, True)
    if len(corners) <= 4:
        break



img_contours = cv2.drawContours(img.copy(), corners, -1 , (0, 0, 255), 3)


img_contours = cv2.polylines(img.copy(), [corners], 
                      True, (0, 0, 255), 20)


mask_largest_form = np.zeros_like(mask)

mask_largest_form = cv2.drawContours(image = mask_largest_form.copy(),  contours= [corners], contourIdx= -1, color= 255, thickness=cv2.FILLED)

mask_largest_form = cv2.dilate(mask_largest_form, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)


y_min, x_min = np.inf, np.inf
y_max, x_max = 0, 0

for corner in corners:
    coord_x, coord_y = corner[0]
    
    if coord_x > x_max:
        x_max = coord_x

    if coord_y > y_max:
        y_max = coord_y
    
    if x_min > coord_x:
        x_min = coord_x
        
    if y_min > coord_y:
        y_min = coord_y

img_cropped = cv2.bitwise_and(img, img, mask = mask_largest_form)

img_resize = np.array(img_cropped[y_min:y_max,x_min:x_max,:])

path_output = "./outputs"

os.makedirs(path_output, exist_ok=True)

cv2.imwrite(f"{path_output}/img_original.jpg", img)
cv2.imwrite(f"{path_output}/img_mask.jpg", mask_largest_form)
cv2.imwrite(f"{path_output}/img_cropped.jpg", img_cropped)
cv2.imwrite(f"{path_output}/img_resize.jpg", img_resize)

cv2.namedWindow("Image Original", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Image Original", 576, 324 ) 
cv2.imshow("Image Original",img)



cv2.namedWindow("Image Gray", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Image Gray", 576, 324 ) 
cv2.imshow("Image Gray",gray_img)

cv2.namedWindow("Image Edges", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Image Edges", 576, 324 ) 
cv2.imshow("Image Edges",edges)

cv2.namedWindow("Mask Plataforma", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Mask Plataforma", 576, 324 ) 
cv2.imshow("Mask Plataforma",mask)

cv2.namedWindow("Mask Largest", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Mask Largest", 576, 324 ) 
cv2.imshow("Mask Largest",mask_largest_form)

cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Image Detection", 576, 324 ) 
cv2.imshow("Image Detection",img_contours)


cv2.namedWindow("Imagen Recortada", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Imagen Recortada", 576, 324 ) 
cv2.imshow("Imagen Recortada",img_cropped)


cv2.namedWindow("Imagen Resize", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Imagen Resize", 576, 324 ) 
cv2.imshow("Imagen Resize",img_resize)



cv2.waitKey(0)