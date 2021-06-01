#####################################################################
#####################################################################
####    SECOM Data Preprocessing Implementation - Version 1.0    ####
#####################################################################
#####################################################################

############################## Update 1.0 ###########################
## 1. 여러 지표상 PASS/FAIL 구분 외에 큰 차이가 없어 리팩토링 진행     ##
## 2. 처리 단계 별로 모듈화, DataAnalytics 에서 캡슐화하여 결과 출력   ##
#####################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def DataAnalytics(file_link):
	pass


def raw_data_refine(raw_data):
	# transpose
	raw_df_stat = raw_data.transpose()

	# 분석을 위해 숫자 데이터만 분리하고 Feature 번호를 만들어줌
	raw_DF_char = raw_data.loc[:, ['Time', 'Pass/Fail']]
	raw_DF_inte = raw_data.drop(['Time', 'Pass/Fail'], axis=1).add_prefix('F')
	raw_DF_original = pd.concat([raw_DF_char, raw_DF_inte], axis=1)

	return raw_DF_original

def close_std_zero_remove(raw_df, num):
	# 이 함수는 표준편차가 num미만인 데이터를 제거합니다.
	raw_df_trans = raw_df.describe().transpose()

	remove_std = raw_df_trans[raw_df_trans['std'] <= num].index
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

def missing_value_refine(filename, data, per):
	# 퍼센트에 따른 결측치 상위 Feature 제거
	if os.path.isfile(filename):
		imputed_data = pd.read_csv(filename)
		imputed_data = imputed_data.drop(imputed_data.describe().columns[0], axis=1)
		return imputed_data
	
	null_var = data.isnull().sum()
	null_df = pd.DataFrame(null_var)
	null_df['null_percentage'] = (null_df[0] / len(null_df.index))

	null_list = null_df[null_df['null_percentage'] > per].index
	deleted_data = data.drop(null_list, axis=1)
	save_cols = list(deleted_data.describe().columns)
	# Missing Value를 다중 대치법으로 채움 -> 옵션제공 예정
	try:
		save_cols = list(deleted_data.describe().columns)
		save_char_df = deleted_data.loc[:, ['Time', 'Pass/Fail']]
		imp_data = deleted_data.drop(['Time', "Pass/Fail"], axis=1)
		print("Impute Start")
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=8, verbose=False).fit_transform(imp_data), columns=save_cols[1:])
		processed_df = pd.concat([save_char_df, imputed_df], axis=1)
		print("Impute End")
	except:
		print("Impute Error - Restart")
		save_cols = list(deleted_data.describe().columns)
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=8, verbose=False).fit_transform(deleted_data), columns=save_cols)
		processed_df = imputed_df
		print("Impute End")

	processed_df.to_csv(filename)
	return processed_df

def outlier_processing(data, per):
	# 이상치를 제거, 보정하는 기능을 합니다.
	# Inter Quantile Range 편차를 이용함
	# 데이터 셋에서 Pass와 Fail을 분리함
	pass_data = data[data['Pass/Fail'] == -1].drop(['Time', 'Pass/Fail'], axis=1)
	fail_data = data[data['Pass/Fail'] == 1].drop(['Time', 'Pass/Fail'], axis=1)
	total_data = data.drop(['Time', 'Pass/Fail'], axis=1)
	# 문자 멈춰!
	col_list = list(data.columns)
	col_list.remove('Pass/Fail')

	pass_corr = correlation_refine(pass_data, 0.5, 0.8)
	fail_corr = correlation_refine(fail_data, 0.5, 0.8)
	#visual = sns.clustermap(pass_corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
	#plt.show()
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






