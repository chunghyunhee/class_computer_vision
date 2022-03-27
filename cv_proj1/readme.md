## Project #1. 

### How to Run 
```shell
## Method1. HoG + resie + bin 조정
$ python src_4_gradient.py

## Method2. HoG 반복실험 + noise 제거 + bin 조정
$ src_5_gradient.ipynb

## Method3. Image Histogram + 최종 결과 시각화
$ python src_2.py

## Other gradient experiments
$ python src_3_gradient.py
```

### Experimental Steps
- step 1. 마우스로 img1, img2에 대해서 모서리 이미지 crop하고 저장 
- step 2. 다운 받은 이미지들 가져와서 히스토그램 그린다음 다시 이미지로 저장
- step 3. 히스토그램 비교하면서 distance 계산하기 (img1의 4개의 점에 대해, img2에 모두의 점만 비교하면됨)
- step 4. distacne 제일 작은것 끼리 선으로 연결지어서 전체 시각화 (src_2에서 최종결과 확인가능)



### Final Result
![result](./repeated_final_result.png)
