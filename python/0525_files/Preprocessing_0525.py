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
	# RAW-DATA : raw-data 가공
	# REFINE1  : 
	# REFINE2  : 변수 간 상관관계를 분석하여, 수치가 높은 Feature를 제거
	## Feature간 상관관계가 높으면 회귀분석이 어려워짐.
	# REFINE3  : 결측치 비율에 따라 일정 비율 이상 Feature 제거 및 보정
	## 결측치 관련 내용은 지난 발표 2 참고. 결측치 처리 알고리즘을 새로운 하나를 더 적용해볼지 고민중
	#[진행완료]---------------------------------------------------------------#

	# REFINE4  : 이상치 비율에 따라 일정 비율 이상 Feature 제거 및 보정
	## 이상탐지 기법은 Clustering과 Classification 기반이 있다.
	## 여기서 나는 이상치를 탐지하기 전에, 효율성을 위해 상관관계 분석을 한번 하고 싶다.
	## (사진 자료) REFINE2 를 거친 데이터들은 그렇게 높은 상관관계가 없으나,
	## (사진 자료) Pass / Fail 을 두개의 데이터 셋으로 나누고, 그 안에서의 상관관계를 확인, 중요 Feature를 알고 싶다.
	
	### 대표적인 Clustering 기반 알고리즘
	### DBSCAN, ROCK, SNN Clustering은 클러스터를 찾고, 그 외의 데이터를 이상치로 처리한다. 
	### K-means, EM-Alogirhtm등은 클러스터의 중심과 데이터 사이의 거리를 점수로 환산, 처리한다.
	### 
	### 대표적인 Classification 기반 알고리즘
	### Neural Network, Bayesian networks, SVM, KNN, LOF
	
	#### (계획) 여러 알고리즘들을 실험, 조사해 보고, 그 중 가장 나은 Clustering 하나, Classification 하나씩 선택.
	#### References
	#### http://docs.iris.tools/manual/IRIS-Usecase/AnomalyDetection/AnomalyDetection_202009_v01.html
	#### https://ko.logpresso.com/documents/anomaly-detection
	#### https://jayhey.github.io/novelty%20detection/2017/10/18/Novelty_detection_overview/
	
	# REFINE5  : 데이터 스케일링 작업
	
	# REFINE6  : Feature Selection (RFE)
	
	# REFINE7  : Oversampling (Pass data)
	
	# REFINE8  : 마지막 단계 (미정) 

	### 0525 피드백으로 인한 계획 수정
	### 1. Pass Fail 데이터를 처음부터 분리한다. (pass, fail, all) - 데이터셋 3개
	### 2. REFINE STEP 1까지는 동일한 과정을 거친다.
	### 3. 상관관계의 비율의 값은 현재 3개에서 2개 혹은 1개로 수정한다.
	### 4. 3번 내용은 결과를 시각화 해서 의미를 파악하도록 하겠습니다.
	### 5. REFINE STEP 2도 3~4과정과 유사하게 진행하겠습니다.
	### 6. 이상치 확인의 경우 STEP 3 이후 고민하겠습니다. 




	#원본 데이터 셋을 받아서 약간의 과정을 거칩니다. 
	raw_DF_original = pd.read_csv(file_link)
	raw_DF_statistic = raw_DF_original.describe().transpose()
	raw_DF_origin = raw_data_refine(raw_DF_original)

	# 가공이 수월한 DF, origin을 생성함.
	print("=============== STEP 0 ==============")
	print(raw_DF_origin.describe())
	print("\n\n")

	# 의미가 없는 표준편차가 0.005 이하인 Feature를 제거, 그리고 데이터 셋 분리
	rf1_df = close_std_zero_remove(raw_DF_origin)
	print("=============== STEP 1 ==============")
	print("\n\n")

	# Feature간 상관 관계 분석을 위해 데이터셋 분리 후 실험
	rf2_main_df = rf1_df;
	rf2_pass_df = rf1_df[rf1_df['Pass/Fail'] == -1]
	rf2_fail_df = rf1_df[rf1_df['Pass/Fail'] == 1]

	# 상관관계 분석, 이후 세 개의 데이터 셋에서의 배타적인 피쳐 분석
	# 세 개의 데이터 셋에서 피쳐의 개수 변화는 Fail만 변화가 있었음.
	# 실행 결과는 주석으로 남기며, 실제 사용시에는 print 문을 모두 주석 처리해야 함.
	rf2_main_c30_df = correlation_refine(rf2_main_df, 0.30, 0.8)
	rf2_pass_c30_df = correlation_refine(rf2_pass_df, 0.30, 0.8)
	rf2_fail_c30_df = correlation_refine(rf2_fail_df, 0.30, 0.8)

	#rf2_main_c60_df = correlation_refine(rf2_main_df, 0.60, 0.8)
	#rf2_pass_c60_df = correlation_refine(rf2_pass_df, 0.60, 0.8)
	#rf2_fail_c60_df = correlation_refine(rf2_fail_df, 0.60, 0.8)

	##### 여기서부터는 주석 처리해도 됨
	rf2_main_c30_idx = list(rf2_main_c30_df.describe().columns)
	rf2_pass_c30_idx = list(rf2_pass_c30_df.describe().columns)
	rf2_fail_c30_idx = list(rf2_fail_c30_df.describe().columns)

	c30_main_pass_same = list(set(rf2_main_c30_idx).intersection(rf2_pass_c30_idx))
	c30_main_fail_same = list(set(rf2_main_c30_idx).intersection(rf2_fail_c30_idx))
	c30_pass_fail_same = list(set(rf2_fail_c30_idx).intersection(rf2_pass_c30_idx))

	####################################################################################
	## | main - pass | main - fail | pass - fail | 3개 유형에서 상호 배타인 피쳐 추출   ##
	####################################################################################
	rf2_mp_c30_main_exclsv = sorted([x for x in rf2_main_c30_idx if x not in rf2_pass_c30_idx])
	rf2_mp_c30_pass_exclsv = sorted([x for x in rf2_pass_c30_idx if x not in rf2_main_c30_idx])
	rf2_mf_c30_main_exclsv = sorted([x for x in rf2_main_c30_idx if x not in rf2_fail_c30_idx])
	rf2_mf_c30_fail_exclsv = sorted([x for x in rf2_fail_c30_idx if x not in rf2_main_c30_idx])
	rf2_pf_c30_pass_exclsv = sorted([x for x in rf2_pass_c30_idx if x not in rf2_fail_c30_idx])
	rf2_pf_c30_fail_exclsv = sorted([x for x in rf2_fail_c30_idx if x not in rf2_pass_c30_idx])

	rf2_c30_exc_dict = {
	'main - pass' : rf2_mp_c30_main_exclsv,
	'main - fail' : rf2_mf_c30_main_exclsv,
	'pass - main' : rf2_mp_c30_pass_exclsv,
	'pass - fail' : rf2_pf_c30_pass_exclsv,
	'fail - main' : rf2_mf_c30_fail_exclsv,
	'fail - pass' : rf2_pf_c30_fail_exclsv
	}
	print(rf2_c30_exc_dict)
	csv_file = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rf2_c30_exc_dict.items()]))
	csv_file.to_csv('./rf2_c30_exclusive_features.csv')
	
	####################################################################################
	##  위 내용에서 main-pass + main-fail 과 같이 구분해 main  상호 배타 피쳐 추출 (3회) ##
	####################################################################################
	rf2_main_exclusive = sorted(list(set(a+c)))
	print(rf2_main_exclusive)
	### FIND 57 EXCLUSIVE FEATURES ###
	#rf2_c30_main_exc_list = ['F136', 'F144', 'F153', 'F155', 'F156', 'F157', 'F173', 'F175', 'F176', 'F182', 'F198', 'F200', 'F21', 'F218', 'F224', 'F247', 'F248', 'F25', 'F251', 'F255', 'F26', 'F269', 'F27', 'F277', 'F286', 'F290', 'F292', 'F301', 'F302', 'F317', 'F318', 'F32', 'F320', 'F33', 'F331', 'F334', 'F34', 'F35', 'F39', 'F426', 'F438', 'F439', 'F45', 'F473', 'F474', 'F476', 'F492', 'F549', 'F550', 'F551', 'F554', 'F560', 'F578', 'F585', 'F66', 'F70', 'F72']

	rf2_pass_exclusive = sorted(list(set(b+d)))
	print(rf2_pass_exclusive)
	### FIND 46 EXCLUSIVE FEATURES ###
	#rf2_c30_pass_exc_list = ['F117', 'F137', 'F139', 'F159', 'F16', 'F166', 'F170', 'F171', 'F184', 'F185', 'F187', 'F201', 'F206', 'F209', 'F221', 'F222', 'F245', 'F252', 'F267', 'F272', 'F273', 'F274', 'F275', 'F291', 'F295', 'F319', 'F342', 'F345', 'F346', 'F347', 'F356', 'F361', 'F382', 'F383', 'F390', 'F393', 'F4', 'F407', 'F478', 'F496', 'F573', 'F575', 'F577', 'F73', 'F74', 'Pass/Fail']
	
	rf2_c30_fail_exclusive = sorted(list(set(c+f)))
	print(rf2_pass_exclusive)
	### FIND 53 EXCLUSIVE FEATURES ###
	#rf2_fail_exc_list = ['F136', 'F153', 'F155', 'F156', 'F157', 'F159', 'F182', 'F198', 'F200', 'F201', 'F21', 'F224', 'F247', 'F248', 'F25', 'F255', 'F26', 'F267', 'F27', 'F273', 'F274', 'F292', 'F302', 'F317', 'F318', 'F319', 'F32', 'F320', 'F33', 'F331', 'F334', 'F34', 'F345', 'F35', 'F356', 'F361', 'F39', 'F426', 'F438', 'F439', 'F45', 'F474', 'F476', 'F492', 'F549', 'F550', 'F551', 'F554', 'F560', 'F578', 'F585', 'F70', 'F72']

	rf2_c30_all_exclusive = sorted(list(set(rf2_main_exclusive + rf2_pass_exclusive + rf2_fail_exclusive)))
	print(rf2_pass_exclusive)
	## FIND 103 Exclusive Features ###
	#rf2_all_exc_list = ['F117', 'F136', 'F137', 'F139', 'F144', 'F153', 'F155', 'F156', 'F157', 'F159', 'F16', 'F166', 'F170', 'F171', 'F173', 'F175', 'F176', 'F182', 'F184', 'F185', 'F187', 'F198', 'F200', 'F201', 'F206', 'F209', 'F21', 'F218', 'F221', 'F222', 'F224', 'F245', 'F247', 'F248', 'F25', 'F251', 'F252', 'F255', 'F26', 'F267', 'F269', 'F27', 'F272', 'F273', 'F274', 'F275', 'F277', 'F286', 'F290', 'F291', 'F292', 'F295', 'F301', 'F302', 'F317', 'F318', 'F319', 'F32', 'F320', 'F33', 'F331', 'F334', 'F34', 'F342', 'F345', 'F346', 'F347', 'F35', 'F356', 'F361', 'F382', 'F383', 'F39', 'F390', 'F393', 'F4', 'F407', 'F426', 'F438', 'F439', 'F45', 'F473', 'F474', 'F476', 'F478', 'F492', 'F496', 'F549', 'F550', 'F551', 'F554', 'F560', 'F573', 'F575', 'F577', 'F578', 'F585', 'F66', 'F70', 'F72', 'F73', 'F74', 'Pass/Fail']


	# 이제 corr_30과 corr_60에서 해당 피쳐들의 DF 추출
	rf2_filt30_main_df = rf2_main_c30_df.transpose().loc[rf2_main_exc_list]
	rf2_filt30_pass_df = rf2_main_c30_df.transpose().loc[rf2_pass_exc_list]
	rf2_filt30_fail_df = rf2_main_c30_df.transpose().loc[rf2_fail_exc_list]
	rf2_filt_all_df = rf2_main_c30_df.transpose().loc[rf2_all_exc_list]

	print(rf2_filt30_main_df)
	print(rf2_filt30_pass_df)
	print(rf2_filt30_fail_df)
	print(rf2_filt_all_df)
	
	#test_corr = csv_file.corr()
	#missing_value_refine('./dd.csv', csv_file, 0.5)
	#visual = sns.clustermap(test_corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
	#plt.show()
	return;
	'''
	데이터 셋 분석 결과, 유의미한 결과가 나왔고, 중복되지 않는 Feature들을 따로 시각화 해보도록 하겠습니다.
	# corr_30 : 기존 437 동일
	[main, pass, fail] 347, 347, 343

	[main-pass] 342 중복 / main 5, pass 5
	[main-fail] = 313 중복 / main 34, fail 31
	[pass-fail] = 312 중복 / pass 35, fail 31
	중복되지 않는 feature들 *(중복포함 141개)

	# corr_60 : 기존 437 동일
	[main, pass, fail] 256, 256, 249

	[main-pass] 237 중복 / main 19, pass 19
	[main-fail] 208 중복 / main 48, fail 41
	[pass-fail] 203 중복 / pass 53, fail 64
	'''

	print("=============== STEP 2 ==============")
	'''
	print("\nmain : ")
	print(rf2_main_c30_df.describe())
	print("\n\npass : ")
	print(rf2_pass_c30_df.describe())
	print("\n\nfail : ")
	print(rf2_fail_c30_df.describe())
	print("\nmain : ")
	print(rf2_main_c60_df.describe())
	print("\n\npass : ")
	print(rf2_pass_c60_df.describe())
	print("\n\nfail : ")
	print(rf2_fail_c60_df.describe())
	'''
	print("\n\n")



	return;
	# 프로젝트 편의를 위해 데이터 저장
	rf2_main_c30_df.to_csv('./rf2_main_c30.csv')
	rf2_pass_c30_df.to_csv('./rf2_pass_c30.csv')
	rf2_fail_c30_df.to_csv('./rf2_fail_c30.csv')
	rf2_main_c60_df.to_csv('./rf2_main_c60.csv')
	rf2_pass_c60_df.to_csv('./rf2_pass_c60.csv')
	rf2_fail_c60_df.to_csv('./rf2_fail_c60.csv')

	#transpose
	rf2_main_c30_df = rf2_main_c30_df.describe()
	rf2_pass_c30_df = rf2_pass_c30_df.describe()
	rf2_fail_c30_df = rf2_fail_c30_df.describe()

	return;
	# 결측치 제거와 보정
	rf3_main_c30_m40 = missing_value_refine("./rf2_main_c25_df.csv", rf2_main_c25_df, 0.4)
	rf3_pass_c30_m40 = missing_value_refine("./rf2_pass_c25_df.csv", rf2_pass_c25_df, 0.4)
	rf3_fail_c30_m40 = missing_value_refine("./rf2_fail_c25_df.csv", rf2_fail_c25_df, 0.4)

	rf3_main_c30_m60 = missing_value_refine("./rf2_main_c25_df.csv", rf2_main_c25_df, 0.6)
	rf3_pass_c30_m60 = missing_value_refine("./rf2_pass_c25_df.csv", rf2_pass_c25_df, 0.6)
	rf3_fail_c30_m60 = missing_value_refine("/rf2_fail_c25_df.csv", rf2_fail_c25_df, 0.6)

	print("=============== STEP 3 ==============")
	print(rf3_main_c25_m40.describe())
	print(rf3_pass_c25_m40.describe())
	print(rf3_fail_c25_m40.describe())
	print()
	print(rf3_main_c25_m60.describe())
	print(rf3_pass_c25_m60.describe())
	print(rf3_fail_c25_m60.describe())

	print("\n\n\n\n")
	
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
	
	print("=============== STEP 4 ==============")
	# 아웃라이어에 대한 정보.


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

	# Missing Value를 다중 대치법으로 채움 -> 옵션제공 예정
	save_cols = ["Time", "Pass/Fail"] + list(deleted_data.describe().columns)[1:]
	save_char_df = deleted_data.loc[:, ['Time', 'Pass/Fail']]
	imp_data = deleted_data.drop(['Time', "Pass/Fail"], axis=1)

	print("Impute Start")
	imputed_df = pd.DataFrame(IterativeImputer(max_iter=8, verbose=False).fit_transform(imp_data), columns=save_cols[2:])
	processed_df = pd.concat([save_char_df, imputed_df], axis=1)
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






