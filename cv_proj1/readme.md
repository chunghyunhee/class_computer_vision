## Project #1. 

### How to Run 
```shell
$ python src_2.py
```

### Experimental Steps
- step 1. 마우스로 img1, img2에 대해서 모서리 이미지 crop하고 저장 
- step 2. 다운 받은 이미지들 가져와서 히스토그램 그린다음 다시 이미지로 저장
- step 3. 히스토그램 비교하면서 distance 계산하기 (img1의 4개의 점에 대해, img2에 모두의 점만 비교하면됨)
- step 4. distacne 제일 작은것 끼리 선으로 연결지어서 전체 시각화


### Results
- Crop한 이미지들: ./cropped_1, ./cropped_2
- Histogram: ./result_histogram_1, ./result_histogram_2
- Distsance Results: result_distance.png
- Final Result: final_result.png
