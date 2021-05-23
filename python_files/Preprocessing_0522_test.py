import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 추후 확장성을 고려하여 제작하였습니다.
# 하나의 함수로 만들어져 있으나, 처리 과정을 분리하여 만들 계획입니다.
# 작업 흐름을 중점적으로 봐주시면 좋겠습니다.

def DataAnalytics(file_link):

	#원본 데이터 셋과 통계용 데이터 셋을 만듭니다. 
	raw_DF_original = pd.read_csv(file_link)
	raw_DF_statistic = raw_DF_original.describe().transpose()



	########## 데이터 구조 분석 과정 시작 ##########


	# 데이터 비율과 통계적 수치 확인
	## 시각화 시작 ##
	#print("PASS / FAIL\n데이터 비율")
	Pass_Fail = raw_DF_original['Pass/Fail'].value_counts().values

	ratio = [Pass_Fail[0]/len(raw_DF_statistic), Pass_Fail[1]/len(raw_DF_statistic)]
	labels = ['Pass', 'Fail']
	wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}


	plt.pie(ratio, labels=labels, autopct='%.1f%%',startangle=260, counterclock=False, wedgeprops=wedgeprops)
	plt.title('Pass/Fail')
	#plt.show()

	## 시각화 종료 ##

	# 데이터 프레임은 시각화가 필요 없으나, 그 내용을 가지고 무언가를 추론할 때 필요 */
	raw_DF_statistic.describe() 
	

	# 분석을 위해 숫자 데이터만 분리하고 Feature 번호를 만들어줌
	raw_DF_char = raw_DF_original.loc[:, ['Time', 'Pass/Fail']]
	raw_DF_inte = raw_DF_original.drop(['Time', 'Pass/Fail'], axis=1).add_prefix('F')
	raw_DF_original = pd.concat([raw_DF_char, raw_DF_inte], axis=1)

	print("=============== RAW-DATA ==============")
	print(raw_DF_original)
	print("\n\n")



	########## 데이터 구조 분석 과정 종료 ##########

	#### 분석한 결과를 토대로 데이터를 가공 시작 ####

	########## 데이터 구조 가공 과정 시작 ##########


	# 통계상 무의미한 Feature 제거 #

	# 표준편차가 0인 Feature.
	# 여기서 전체 데이터와 표준편차가 0인 데이터를 시각화 하고
	# 가공 과정 단계마다 처리 과정 전 후의 차이를 시각화 데이터를 만들어 두면 좋을듯.
	raw_DF_statistic = raw_DF_statistic.describe().transpose()
	remove_std = raw_DF_statistic[raw_DF_statistic['std'] == 0].index

	# 표준편차 제거가 이루어진 데이터
	refine1_DF = raw_DF_original.drop(remove_std)
	print("=============== REFINE 1 ==============")
	print(refine1_DF.describe())
	print("\n\n")


	# 표준편차가 0에 가까운 Feature 확인
	refine1_DF_statistic = refine1_DF.describe().transpose()
	remove_std = refine1_DF_statistic[(refine1_DF_statistic['std'] > 0) & (refine1_DF_statistic['std'] < 0.005)].index

	# 과정 이해를 돕기 위해 변수를 남발하고 있습니다.
	# 최종 발표 시점에는 리팩토링을 진행하도록 하겠습니다.
	# 주석도 깔끔히 정리할 예정입니다... 
	refine2_DF = refine1_DF.drop(remove_std, axis=1)
	print("=============== REFINE 2 ==============")
	print(refine2_DF.describe())
	print("\n\n")


	# 변수 간 상관관계, 다중 공선성... 
	## abs 8 의 기준 ? (수정해야 함)
	corr_data = refine2_DF.corr()
	corr_data_nan = corr_data[corr_data > abs(0.8)]

	# 다중 공선성이 높은 Feature를 찾는 과정
	col_names = list(corr_data_nan)
	row_names = list(corr_data_nan.index)
	corr_list = []

	for i in range(0, len(col_names)):
		for j in range(0, len(row_names)):
			temp = []
			if(corr_data_nan[col_names[i]][row_names[j]] > 0.8):
				temp.append(col_names[i])
				temp.append(row_names[j])
				temp.append(corr_data_nan[col_names[i]][row_names[j]])
				corr_list.append(temp)


	# 찾은 내용을 지우도록 합니다. 기존 계획과 약간 순서가 달라졌습니다.
	# 이 부분 전 후로 데이터셋의 변화를 시각화 해주시면 됩니다.
	# 다중공선성에 대한 기준은 추후 수정하겠지만, 전 후 상황을 비교할 수 있는 자료를 부탁드립니다.
	corr_list_df = pd.DataFrame(corr_list)
	corr_result = corr_list_df.drop_duplicates([2], keep="first")

	x = corr_result[0].value_counts()
	xdf = pd.DataFrame(x)
	y = corr_result[1].value_counts()
	ydf = pd.DataFrame(y)

	# 다중 공선성이 높은 데이터들이 있는 데이터 프레임
	## 기존 방법에서는 이 프레임의 결측치를 0으로 채우고 진행하였음.
	### 문제가 발생할 여지가 있으니 유의할 것.
	corr_df = pd.concat([xdf, ydf], ignore_index=True, axis=1)
	corr_pc = corr_df.fillna(0)
	corr_df['sum'] = corr_pc[0]+corr_pc[1]
	corr_df = corr_df.sort_values(by=['sum'], axis=0, ascending=False)
	
	#통계를 살펴본 결과, 중간값이 2, 평균이 3 정도... 
	#print(corr_df.describe())

	extract_20 = []
	extract_30 = []
	extract_40 = []

	# 다중 공선성이 높은 상위 20%, 30%, 40% 개 제거
	for i in range(0, int(len(corr_df.index)*0.2)):
		extract_20.append(list(corr_df.index)[i])

	for i in range(0, int(len(corr_df.index)*0.3)):
		extract_30.append(list(corr_df.index)[i])

	for i in range(0, int(len(corr_df.index)*0.4)):
		extract_40.append(list(corr_df.index)[i])

	# 정제된 3가지 버전의 DataFrame
	refine3_ex20 = refine2_DF.drop(extract_20, axis=1)
	refine3_ex30 = refine2_DF.drop(extract_30, axis=1)
	refine3_ex40 = refine2_DF.drop(extract_40, axis=1) 
	
	print("\n")
	print("=============== REFINE 3 ==============")
	print("============== EXTRACT 20 =============")
	print(refine3_ex20)
	print("\n")
	print("============== EXTRACT 30 =============")
	print(refine3_ex30)
	print("\n")
	print("============== EXTRACT 40 =============")
	print(refine3_ex40)
	print("\n\n\n")


	# 결측치 처리
	## 결측치가 ~40%인 데이터셋, ~50%인 데이터셋, ~60%인 데이터셋으로 분리
	refine3_ex20_trans = refine3_ex20.transpose()
	refine3_ex30_trans = refine3_ex30.transpose()
	refine3_ex40_trans = refine3_ex40.transpose()

	# 결측치 비율 확인 < 버그 >
	null_ex20 = refine3_ex20_trans.isnull().sum()
	null_ex30 = refine3_ex30_trans.isnull().sum()
	null_ex40 = refine3_ex40_trans.isnull().sum()

	print(null_ex20)
	print(null_ex30)
	print(null_ex40)

	return;
	null_df = pd.DataFrame(rf3_null_stat)
	null_df['null_percent'] = (null_df[0] / len(null_df.index)) * 100
	print(null_df)

	rm_40 = null_df[null_df['null_percent'] > 0.4].index
	rm_50 = null_df[null_df['null_percent'] > 0.5].index
	rm_60 = null_df[null_df['null_percent'] > 0.6].index

	refine4_rm40 = refine3_DF.drop(rm_40, axis=1)
	refine4_rm50 = refine3_DF.drop(rm_50, axis=1)
	refine4_rm60 = refine3_DF.drop(rm_60, axis=1)

	print()
	print(refine4_rm40)
	print()
	print(refine4_rm50)
	print()
	print(refine4_rm60)
	print()

DataAnalytics('./uci-secom.csv')






