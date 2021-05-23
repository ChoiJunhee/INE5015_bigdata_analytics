import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# 데이터 전처리에 있어 여러 결과를 도출하기 위한 
# 기본적인 기능을 모듈화 하여 프로그래밍 하였습니다.
# 이제 최적의 전처리 방법을 통해 최적의 데이터셋을 얻으면 됩니다.

def DataAnalytics(file_link):
	# RAW-DATA : 원본 데이터에서 필요한 데이터만 분리.
	# REFINE1  : 의미가 없는 표준편차가 0.005 이하인 Feature를 제거
	# REFINE2  : 변수 간 상관관계를 분석하여, 수치가 높은 Feature를 제거
	# REFINE3  : 결측치 비율에 따라 일정 비율 이상 Feature 제거 및 보정
	# REFINE4  : 이상치 비율에 따라 일정 비율 이상 Feature 제거 및 보정
	# REFINE5  : 데이터 스케일링 작업
	# REFINE6  : Feature Selection (RFE)
	# REFINE7  : Oversampling (Pass data)
	# REFINE8  : 마지막 단계 (미정) 

	#원본 데이터 셋과 통계용 데이터 셋을 만듭니다. 
	raw_DF_original = pd.read_csv(file_link)
	raw_DF_statistic = raw_DF_original.describe().transpose()


	raw_DF_origin = raw_data_refine(raw_DF_original)
	print("=============== RAW-DATA ==============")
	print(raw_DF_origin.describe())
	print("\n\n")


	refine1_DF = close_std_zero_remove(raw_DF_origin)
	print("=============== REFINE 1 ==============")
	print(refine1_DF.describe())
	print("\n\n")

	# @param : DataFrame, percentage, abs_num
	# 상관관계가 높은 Feature 제거
	refine2_ex20 = correlation_refine(refine1_DF, 0.2, 0.8)
	refine2_ex30 = correlation_refine(refine1_DF, 0.3, 0.8)
	refine2_ex40 = correlation_refine(refine1_DF, 0.4, 0.8)
	print("=============== REFINE 2 ==============")
	print("[corr 상위 20% 제거]")
	print(refine2_ex20.describe())
	print()
	print("[corr 상위 30% 제거]")
	print(refine2_ex30.describe())
	print()
	print("[corr 상위 40% 제거]")
	print(refine2_ex40.describe())
	print("\n\n")



	# 결측치 제거 (이 부분도 리팩토링 하도록 하겠습니다.)
	# 결측치 보정 과정에서 시간이 너무 오래 걸려 계산해둔 파일을 쓰는 방식으로 변경
	refine3_ex20_nl40 = missing_value_refine("rf3_e20_n40", refine2_ex20, 0.4)
	refine3_ex20_nl50 = missing_value_refine("rf3_e20_n50", refine2_ex20, 0.5)
	refine3_ex20_nl60 = missing_value_refine("rf3_e20_n60", refine2_ex20, 0.6)
	
	refine3_ex30_nl40 = missing_value_refine("rf3_e30_n40", refine2_ex30, 0.4)
	refine3_ex30_nl50 = missing_value_refine("rf3_e30_n50", refine2_ex30, 0.5)
	refine3_ex30_nl60 = missing_value_refine("rf3_e30_n60", refine2_ex30, 0.6)
	
	refine3_ex40_nl40 = missing_value_refine("rf3_e40_n40", refine2_ex40, 0.4)
	refine3_ex40_nl50 = missing_value_refine("rf3_e40_n50", refine2_ex40, 0.5)
	refine3_ex40_nl60 = missing_value_refine("rf3_e40_n60", refine2_ex40, 0.6)

	refine3_nl40_list = [refine3_ex20_nl40, refine3_ex30_nl40, refine3_ex40_nl40]
	refine3_nl50_list = [refine3_ex20_nl50, refine3_ex30_nl50, refine3_ex40_nl50]
	refine3_nl60_list = [refine3_ex20_nl60, refine3_ex30_nl60, refine3_ex40_nl60]

	print("=============== REFINE 3 ==============")
	print("[결측치 40%]")
	for i in refine3_nl40_list:
		print(i.describe())
		print()
	print("\n\n")
	print("[결측치 50%]")
	for i in refine3_nl50_list:
		print(i.describe())
		print()

	print("\n\n")
	print("[결측치 60%]")
	for i in refine3_nl60_list:
		print(i.describe())
		print()

	print("\n\n\n")

	# 결측치 제거 부분에서 전혀 처리가 안되는 부분이 있음!


	# outlier 확인
	refine4_nl40_list = []
	refine4_nl50_list = []
	refine4_nl60_list = []

	for i in refine3_nl40_list:
		refine4_nl40_list.append(outlier_processing(i, 0))
	for i in refine3_nl50_list:
		refine4_nl50_list.append(outlier_processing(i, 0))
	for i in refine3_nl60_list:
		refine4_nl60_list.append(outlier_processing(i, 0))
	print("=============== REFINE 4 ==============")

def raw_data_refine(raw_data):
	# transpose
	raw_df_stat = raw_data.transpose()

	# 분석을 위해 숫자 데이터만 분리하고 Feature 번호를 만들어줌
	raw_DF_char = raw_data.loc[:, ['Time', 'Pass/Fail']]
	raw_DF_inte = raw_data.drop(['Time', 'Pass/Fail'], axis=1).add_prefix('F')
	raw_DF_original = pd.concat([raw_DF_char, raw_DF_inte], axis=1)

	return raw_DF_original

def close_std_zero_remove(raw_df):
	# 이 함수는 표준편차가 0.005 미만인 데이터를 제거합니다.
	raw_df_trans = raw_df.describe().transpose()

	remove_std = raw_df_trans[raw_df_trans['std'] <= 0.005].index
	result = raw_df.drop(remove_std, axis=1)

	return result

def correlation_refine(data, per, abs_num):
	corr_data = data.corr()
	corr_data_nan = corr_data[corr_data > abs(abs_num)]

	col_names = list(corr_data_nan)
	row_names = list(corr_data_nan.index)
	corr_list = []

	# 상관관계가 높은 Feature를 찾는 과정
	for i in range(0, len(col_names)):
		for j in range(0, len(row_names)):
			temp = []
			if(corr_data_nan[col_names[i]][row_names[j]] > 0.8):
				temp.append(col_names[i])
				temp.append(row_names[j])
				temp.append(corr_data_nan[col_names[i]][row_names[j]])
				corr_list.append(temp)
	
	corr_list_df = pd.DataFrame(corr_list)
	corr_result = corr_list_df.drop_duplicates([2], keep="first")

	x = corr_result[0].value_counts()
	xdf = pd.DataFrame(x)
	y = corr_result[1].value_counts()
	ydf = pd.DataFrame(y)

	# 다중 공선성이 높은 데이터들이 있는 데이터 프레임
	corr_df = pd.concat([xdf, ydf], ignore_index=True, axis=1)
	corr_pc = corr_df.fillna(0)
	corr_df['sum'] = corr_pc[0]+corr_pc[1]
	corr_df = corr_df.sort_values(by=['sum'], axis=0, ascending=False)

	extract = []
	for i in range(0, int(len(corr_df.index) * per)):
		extract.append(list(corr_df.index)[i])

	result = data.drop(extract, axis=1)
	return result

def missing_value_refine(name, data, per):
	# 퍼센트에 따른 결측치 상위 Feature 제거
	path = './' + str(name)
	filename = name + str('.csv')
	if os.path.isfile(path):
		imputed_data = pd.read_csv(path)
		return imputed_data
	null_var = data.isnull().sum()
	null_df = pd.DataFrame(null_var)
	null_df['null_percentage'] = (null_df[0] / len(null_df.index))

	null_list = null_df[null_df['null_percentage'] > per].index
	deleted_data = data.drop(null_list, axis=1)

	# Missing Value를 다중 대치법으로 채움 -> 옵션제공 예정
	deleted_data = deleted_data.drop(['Time'], axis=1)
	print("Impute Start")
	result = pd.DataFrame(IterativeImputer(verbose=False).fit_transform(deleted_data))
	print("Impute End")
	result.to_csv(filename, index=False)
	return result

def outlier_processing(data, per):
	# 이상치를 제거, 보정하는 기능을 합니다.

	# 데이터 셋에서 Pass와 Fail을 분리함
	pass_data = data[data['Pass/Fail'] == -1]
	fail_data = data[data['Pass/Fail'] == 1]

	# 문자 멈춰!
	col_list = list(data.columns)
	col_list.remove('Pass/Fail')

	#시각화 자료가 필요합니다.

	#IQR Return

def outlier_refine(data, per):
	#IQR을 기반으로 일정 percent가 넘는 이상치를 가지는 Feature를 제거 및 보정
	pass

def data_scaling(data, num):
	# number에 맞는 스케일링을 시작
	pass

def feature_selection(data, num):
	# number에 맞는 Feature Selection을 진행 후 결과 리턴
	pass

def data_oversampling(data, num):
	# number에 맞는 over sampling을 진행 후 결과 리턴
	pass


# 이건 데이터 모델링 과정에서 할 것
def data_modeling():
	# 부스팅, 베깅 등등
	# 하이퍼 파라미터 튜닝
	# gridSearchCV() 등 여러 기능 활용
	# 테스트 데이터셋 제작
	# 검증
	pass



DataAnalytics('./uci-secom.csv')






