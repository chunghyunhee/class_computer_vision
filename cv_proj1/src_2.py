import cv2
import os
import math
from matplotlib import pyplot as plt
import numpy as np

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
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse1)
key = cv2.waitKey(0)
print(key)
cv2.destroyAllWindows()

isDragging_2 = False
x0_2, y0_2, w_2, h_2 = -1, -1, -1, -1
blue_2, red_2 = (255, 0, 0), (0, 0, 255)
count2 = 0

img2 = cv2.imread('2nd.jpg')
cv2.imshow('img2', img2)
cv2.setMouseCallback('img2', onMouse2)
key2 = cv2.waitKey(0)
print(key2)
cv2.destroyAllWindows()

# step2. 다운 받은 이미지들 가져와서 히스토그램 그린다음 다시 이미지로 저장

# img1의 4개의 이미지에 대해서 접근후 히스토그램 생성
for file_nm in os.listdir('cropped_1') :

    if '.DS' in file_nm : 
        continue
    else :
        file_path = os.path.join('cropped_1', file_nm)
        print('-- cropped_1 file path : ', file_path)

        # 이미지 읽고
        img = cv2.imread(file_path,0)

        # 히스토그램 생성하고
        plt.title('Img1 ' + file_nm.split('.')[0])
        plt.hist(img.ravel(), 16, [0,256]); plt.show()
        #hist = cv2.calcHist([img], None, None, [16], [0, 256])
        #plt.plot(hist); plt.show()

        # 저장하기 (histogram_1)
        # plt.savefig(f'histogram_1/hist1_{file_nm}')

# img2의 4개의 이미지에 대해서 접근후 히스토그램 생성
for file_nm in os.listdir('cropped_2') :

    if '.DS' in file_nm :
        continue
    else :
        file_path = os.path.join('cropped_2', file_nm)
        print('-- cropped_2 file path : ', file_path)

        # 이미지 읽고
        img = cv2.imread(file_path,0)

        # 히스토그램 생성하고
        plt.title('Img2 ' + file_nm.split('.')[0])
        plt.hist(img.ravel(), 16, [0,256]); plt.show()
        #hist = cv2.calcHist([img], None , None, [16], [0, 256])
        #plt.plot(hist); plt.show()

        # 저장하기 (histogram_2)
        #plt.savefig(f'histogram_2/hist2_{file_nm}')




# step3. 히스토그램 비교하면서 distance 계산하기 (img1의 4개의 점에 대해, img2에 모두의 점만 비교하면됨)
final_result = dict()

for file_nm_1 in os.listdir('./cropped_1') :
    tmp_result = dict()

    if '.DS' in file_nm_1 : 
        continue
    else :

        file_path_1 = os.path.join('./cropped_1', file_nm_1)
        img1 = cv2.imread(file_path_1)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img1 = clahe.apply(img1)
        img1 = cv2.resize(img1,(400,400))

        hist1 = cv2.calcHist([img1], [0], None, [16], [0, 256]) # sol0

        for file_nm_2 in os.listdir('./cropped_2') :

            if '.DS' in file_nm_2 :
                continue
            else : 
                file_path_2 = os.path.join('./cropped_2', file_nm_2)
                img2 = cv2.imread(file_path_2)    
                img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  # sol1
                img2 = clahe.apply(img2)          # sol2
                img2 = cv2.resize(img2,(400,400)) # sol3 

                hist2 = cv2.calcHist([img2], [0], None, [16], [0, 256])

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



