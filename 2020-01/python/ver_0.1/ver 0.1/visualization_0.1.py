####################################################
####################################################
####  CSV FILE - VISUALIZATION - Version 1.0    ####
####################################################
####################################################
####                                            ####
####################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

# @param file_link - 파일 링크 리스트
# @param opt - 출력할 그래프
# @param x, y - label
def visualization(file_link, opt, x, y):
	data = pd.read_csv(file_link)

	if(1 == opt):
		# Box Flot
		sns.boxplot(data=data, palette="Paired")
		plt.show()
		plt.xticks(rotation=45)
		sns.boxplot(data=data, x=x, y=y, pallette="Paired")
		plt.show()
		
	elif(2 == opt):
		# Histogram
		sns.distplot(data[0][0], kde=False, rug=True)
		## 고쳐야함

	elif(3 == opt):
		# countplot
		plt.xticks(rotation=45)
		sns.countplot(data=data, x=x, y=y)
		plt.show()
		#sns.catplot

	elif(4 == opt):
		sns.relplot(data=data, x=x, y=y)
		plt.show()
		
	elif(5 == opt):
		#Line Plot
		sns.lineplot(data=data, x=x, y=y)
		#피쳐 별 비교 필요
		# relplot 필요
	elif(6 == opt):
		sns.barplot(data=data, x=x, y=y)
		# hue 사용하여 하나의 변수에 대해 나눌 것
		# catplot을 사용해 all, pass, fail 비교해볼 것
		# barplot 은 뭐 쓸 일 없을거같긴한데...
		# heatmap 추가해야함
		## 회귀 분석 그래프를 그리는 Implot 사용해볼것...
		## https://tariat.tistory.com/792
		# 후반에 joinplot 을 통해 1:1 비교 할 수 있게
		# pairplot도 써보고...
		##


visualization('./step4 - missing value/step4_m3_fail.csv', 1, "x", "y")