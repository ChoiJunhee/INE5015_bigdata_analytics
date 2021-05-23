import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 전처리에 있어 여러 결과를 도출하기 위한 
# 기본적인 기능을 모듈화 하여 프로그래밍 하였습니다.
# 이제 최적의 전처리 방법을 통해 최적의 데이터셋을 얻으면 됩니다.

def DataAnalytics(file_link):
	# RAW-DATA : 원본 데이터에서 필요한 데이터만 분리하고, Feature에 이름을 붙여줌
	# REFINE1  : 의미가 없는 표준편차가 0.005 이하인 Feature를 제거함
	# REFINE2  : 변수 간 상관관계를 분석하여, 수치가 높은 Feature를 제거함

	#원본 데이터 셋과 통계용 데이터 셋을 만듭니다. 
	raw_DF_original = pd.read_csv(file_link)
	raw_DF_statistic = raw_DF_original.describe().transpose()


	raw_DF_origin = raw_data_refine(raw_DF_original)
	print("=============== RAW-DATA ==============")
	print(raw_DF_origin.describe())
	print("\n\n")


	refine1_DF = close_std_zero_remove(raw_DF_original)
	print("=============== REFINE 1 ==============")
	print(refine1_DF.describe())
	print("\n\n")

	# @param : DataFrame, percentage, abs_num
	# 상관관계가 높은 Feature 제거
	refine2_ex20 = correlation_refine(refine1_DF, 0.2, 0.8)
	refine2_ex30 = correlation_refine(refine1_DF, 0.3, 0.8)
	refine2_ex40 = correlation_refine(refine1_DF, 0.4, 0.8)
	print("=============== REFINE 2 ==============")
	print("[상위 20% 제거]")
	print(refine2_ex20.describe())
	print()
	print("[상위 30% 제거]")
	print(refine2_ex30.describe())
	print()
	print("[상위 40% 제거]")
	print(refine2_ex40.describe())
	print("\n\n")



	# 결측치 제거 (이 부분도 리팩토링 하도록 하겠습니다.)
	refine3_ex20_nl40 = missing_value_refine(refine2_ex20, 0.4)
	refine3_ex20_nl50 = missing_value_refine(refine2_ex20, 0.5)
	refine3_ex20_nl60 = missing_value_refine(refine2_ex20, 0.6)
	
	refine3_ex30_nl40 = missing_value_refine(refine2_ex30, 0.4)
	refine3_ex30_nl50 = missing_value_refine(refine2_ex30, 0.5)
	refine3_ex30_nl60 = missing_value_refine(refine2_ex30, 0.6)
	
	refine3_ex40_nl40 = missing_value_refine(refine2_ex40, 0.4)
	refine3_ex40_nl50 = missing_value_refine(refine2_ex40, 0.5)
	refine3_ex40_nl60 = missing_value_refine(refine2_ex40, 0.6)

	refine3_nl40_list = [refine3_ex20_nl40, refine3_ex30_nl40, refine3_ex40_nl40]
	refine3_nl50_list = [refine3_ex20_nl50, refine3_ex30_nl50, refine3_ex40_nl50]
	refine3_nl60_list = [refine3_ex20_nl60, refine3_ex30_nl60, refine3_ex40_nl60]

	print("=============== REFINE 4 ==============")
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

def missing_value_refine(data, per):
	# 퍼센트에 따른 결측치 상위 Feature 제거
	null_var = data.isnull().sum()
	null_df = pd.DataFrame(null_var)
	null_df['null_percentage'] = (null_df[0] / len(null_df.index))

	null_list = null_df[null_df['null_percentage'] > per].index
	result = data.drop(null_list, axis=1)

	return result

DataAnalytics('./uci-secom.csv')






