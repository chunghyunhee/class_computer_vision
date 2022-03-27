import cv2
import os
import math
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
# from histogram import *



isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)
count = 0

img1_point_list = list()
img2_point_list = list()


# step1. 마우스로 img1, img2에 대해서 모서리 이미지 crop하고 저장 
def onMouse1(event, x, y, flags, param):
    global isDragging, x0, y0, img, count

    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow('img', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
            if w > 0 and h > 0:
                img_draw = img.copy()
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                img1_point_list.append([x0, y0, x, y])
                cv2.imshow('img', img_draw)
                roi = img[y0:y0+h, x0:x0+w]
                # cv2.imshow('cropped', roi)
                # cv2.moveWindow('cropped', 0, 0)
                # cv2.imwrite(f'./cropped_1/{count}_cropped.png', roi)
                count += 1
            else:
                cv2.imshow('img', img)
                print('drag should start from left-top side')

def onMouse2(event, x, y, flags, param):
    global isDragging_2, x0_2, y0_2, img2, count2

    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging_2 = True
        x0_2 = x
        y0_2 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging_2:
            img_draw = img2.copy()
            cv2.rectangle(img_draw, (x0_2, y0_2), (x, y), blue, 2)
            cv2.imshow('img2', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging_2:
            isDragging_2 = False
            w = x - x0_2
            h = y - y0_2
            if w > 0 and h > 0:
                img_draw = img2.copy()
                cv2.rectangle(img_draw, (x0_2, y0_2), (x, y), red, 2)
                img2_point_list.append([x0_2, y0_2, x, y])
                cv2.imshow('img2', img_draw)
                roi = img2[y0_2:y0_2+h, x0_2:x0_2+w]
                # cv2.imshow('cropped', roi)
                # cv2.moveWindow('cropped', 0, 0)
                # cv2.imwrite(f'./cropped_2/{count2}_cropped.png', roi)

                count2 += 1
            else:
                cv2.imshow('img2', img)
                print('drag should start from left-top side')
 
img = cv2.imread('1st.jpg')
# cv2.imshow('img', img)
# cv2.setMouseCallback('img', onMouse1)
# key = cv2.waitKey(0)
# print(key)
# cv2.destroyAllWindows()

isDragging_2 = False
x0_2, y0_2, w_2, h_2 = -1, -1, -1, -1
blue_2, red_2 = (255, 0, 0), (0, 0, 255)
count2 = 0

img2 = cv2.imread('2nd.jpg')
# cv2.imshow('img2', img2)
# cv2.setMouseCallback('img2', onMouse2)
# key2 = cv2.waitKey(0)
# print(key2)
# cv2.destroyAllWindows()



# step3. 히스토그램 비교하면서 distance 계산하기 (img1의 4개의 점에 대해, img2에 모두의 점만 비교하면됨)
final_result = dict()

for file_nm_1 in os.listdir('./cropped_1') :
    tmp_result = dict()

    if '.DS' in file_nm_1 : 
        continue
    else :

        file_path_1 = os.path.join('./cropped_1', file_nm_1)
        img1 = cv2.imread(file_path_1, 0)
        print(f'{file_nm_1} shape :', img1.shape)


        img1 = imread(file_path_1)
        fd, hog_image = hog(img1, orientations=9, pixels_per_cell=(11, 11),cells_per_block=(7, 7), visualize=True, multichannel=True, block_norm='L2')
            
        # plt.axis("off")
        # plt.imshow(hog_image, cmap="gray")
        # plt.show()

        hist1 = cv2.calcHist([np.float32(hog_image)], [0], None, [16], [0, 256])
        # plt.hist(hist1, 16, [0,16]); plt.show()
        # plt.bar(list(len(fd)), fd)
        print(fd.shape)
        
        
        # img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        # img1 = cv2.equalizeHist(img1)
        # img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        # img1 = cv2.bilateralFilter(img1, -1, 10, 5)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img1 = clahe.apply(img1)
        # img1 = cv2.GaussianBlur(img1, (5,5), 0)
        # img1 = cv2.resize(img1,(400,400))

        # img1_gradient = cv2.Laplacian(img1, cv2.CV_8U, ksize = 11)
        # img1_gradient = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=15)
        # mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

        # gaussian = cv2.GaussianBlur(img1, (15, 15), 0) 
        # LoG_1 = cv2.filter2D(gaussian, -1, mask3)

        # hist1 = cv2.calcHist([np.float32(LoG_1)], [0], None, [16], [0, 255]) 
        # plt.title('Img1 ' + file_nm_1.split('.')[0])
        # plt.hist(hist1, 16, [0,256]); plt.show()

        for file_nm_2 in os.listdir('./cropped_2') :

            if '.DS' in file_nm_2 :
                continue
            else : 
                file_path_2 = os.path.join('./cropped_2', file_nm_2)
                img2 = cv2.imread(file_path_2, 0)
                img2 = imread(file_path_2)
                fd, hog_image_2 = hog(img2, orientations=9, pixels_per_cell=(11, 11),cells_per_block=(7, 7), visualize=True, multichannel=True, block_norm='L2')

                # plt.axis("off")
                # plt.imshow(hog_image_2, cmap="gray")
                # plt.show()

                hist2 = cv2.calcHist([np.float32(hog_image_2)], [0], None, [16], [0, 256])
                # plt.hist(hist2, 16, [0,256]); plt.show()
                # plt.bar(list(len(fd)), fd)
                print(fd.shape)
            

                # img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  # sol1
                # img2 = cv2.bilateralFilter(img2, -1, 10, 5)
                # img2 = cv2.equalizeHist(img2)  
                # img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
                # img2 = clahe.apply(img2)          # sol2
                # img2 = cv2.GaussianBlur(img2, (5,5), 0)
                # img2 = cv2.resize(img2,(400,400)) # sol3 

                # img2_gradient = cv2.Laplacian(img2, cv2.CV_8U, ksize = 11)
                # img2_gradient = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=15)
                # gaussian = cv2.GaussianBlur(img2, (15, 15), 0) 
                # LoG_2 = cv2.filter2D(gaussian, -1, mask3)

                # hist2 = cv2.calcHist([np.float32(LoG_2)], [0], None, [16], [0, 256])
                # plt.title('Img2 ' + file_nm_2.split('.')[0])
                # plt.hist(hist2.ravel(), 16, [0,256]); plt.show()

                # distance 연산
                sum = 0
                for i in range(0, 16) :
                    sum = sum + (hist1[i][0]-hist2[i][0]) ** 2

                dist = math.sqrt(sum)
                #print(f'-- Euclidean distance between {file_nm_1} and {file_nm_2} : ', dist)

                tmp_result[file_nm_2] = dist
    
    # img1의 각 점에서 뭐가 매칭되는지 확인
    print('#######################################')
    tmp_result_sort = sorted(tmp_result.items(), key = lambda item: item[1], reverse = False)
    print(f'-- {file_nm_1} : ', tmp_result_sort )
    print('-- closest point : ', tmp_result_sort[0])

    final_result[file_nm_1] = tmp_result_sort[0][0]
    print()

print('## Final result :', final_result)


# step4. distacne 제일 작은것 끼리 선으로 연결지어서 전체 시각화

## 이미지 2개 띄우고
img1 = cv2.imread('1st.jpg')
img2 = cv2.imread('2nd.jpg')

## 마우스로 crop한 박스 부분 띄우고
blue = (255, 0, 0)
for i in range(4) : 
    img1 = cv2.rectangle(img1, (img1_point_list[i][0], img1_point_list[i][1]), (img1_point_list[i][2], img1_point_list[i][3]), blue, 2)

for i in range(4) : 
    img2 = cv2.rectangle(img2, (img2_point_list[i][0], img2_point_list[i][1]), (img2_point_list[i][2], img2_point_list[i][3]), blue, 2)

addh = cv2.hconcat([img1, img2])
cv2.imshow('concated', addh)


## distance 제일 작은 것끼리 긋기 (tmp_result_sort에서 구한 기준)

for key, value in final_result.items() :
    src_idx = int(key.split('_')[0])
    dst_idx = int(value.split('_')[0])

    #addh_draw = addh.copy()
    addh = addh.astype(np.uint8)
    src_cor = [ ( img1_point_list[src_idx][0] + img1_point_list[src_idx][2] ) /2 , ( img1_point_list[src_idx][1] + img1_point_list[src_idx][3] ) /2 ]
    dst_cor = [ ( (img2_point_list[dst_idx][0] + addh.shape[1]/2) + (img2_point_list[dst_idx][2] + addh.shape[0]/2) ) / 2 + 560,  (img2_point_list[dst_idx][1] + img2_point_list[dst_idx][3] ) / 2 ]

    print(src_cor)
    print(dst_cor)
    # 해당하는 부위끼리 선 긋기
    addh = cv2.line(addh, (int(src_cor[0]), int(src_cor[1])), (int(dst_cor[0]), int(dst_cor[1])), blue, 3)

cv2.imshow('Results', addh)


## 끝내기
cv2.waitKey(0)
cv2.destroyAllWindows()



