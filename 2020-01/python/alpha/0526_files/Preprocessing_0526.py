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
	print(rf1_df.describe())
	print("\n\n")

	# Feature간 상관 관계 분석을 위해 데이터셋 분리 후 실험
	rf2_main_df = rf1_df;
	rf2_pass_df = rf1_df[rf1_df['Pass/Fail'] == -1]
	rf2_fail_df = rf1_df[rf1_df['Pass/Fail'] == 1]

	# 상관관계 분석, 이후 세 개의 데이터 셋에서의 배타적인 피쳐 분석
	# 세 개의 데이터 셋에서 피쳐의 개수 변화는 Fail만 변화가 있었음.
	# 실행 결과는 주석으로 남기며, 실제 사용시에는 print 문을 모두 주석 처리해야 함.
	rf2_c30_main_df = correlation_refine(rf2_main_df, 0.30, 0.8)
	rf2_c30_pass_df = correlation_refine(rf2_pass_df, 0.30, 0.8)
	rf2_c30_fail_df = correlation_refine(rf2_fail_df, 0.30, 0.8)

	'''
	####################################################################################
	## | 데이터 분석 결과, 유의미한 편차가 있었고, 상호배타성 피쳐를 추출하는 과정입니다.   ##
	####################################################################################

	----------------------------------------------------------
	# corr_30 : 기존 437 동일
	[main, pass, fail] 347, 347, 343

	[main-pass] 342 중복 / main 5, pass 5
	[main-fail] = 313 중복 / main 34, fail 31
	[pass-fail] = 312 중복 / pass 35, fail 31
	중복되지 않는 feature들 *(중복포함 141개)
	----------------------------------------------------------
	# corr_60 : 기존 437 동일
	[main, pass, fail] 256, 256, 249

	[main-pass] 237 중복 / main 19, pass 19
	[main-fail] 208 중복 / main 48, fail 41
	[pass-fail] 203 중복 / pass 53, fail 64
	----------------------------------------------------------
	corr 상위 30%과 60%에도 차이가 있었으나, 큰 차이가 없어 30% 사용
	----------------------------------------------------------
	

	rf2_main_c30_idx = list(rf2_c30_main_df.describe().columns)
	rf2_pass_c30_idx = list(rf2_c30_pass_df.describe().columns)
	rf2_fail_c30_idx = list(rf2_c30_fail_df.describe().columns)

	c30_main_pass_same = list(set(rf2_c30_main_df).intersection(rf2_c30_pass_df))
	c30_main_fail_same = list(set(rf2_c30_main_df).intersection(rf2_c30_fail_df))
	c30_pass_fail_same = list(set(rf2_c30_fail_df).intersection(rf2_c30_pass_df))

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
	csv_file.to_csv('./rf2_c30_exc10.csv')
	
	####################################################################################
	##  위 내용에서 main-pass + main-fail 과 같이 구분해 main  상호 배타 피쳐 추출 (3회) ##
	####################################################################################
	rf2_main_exclusive = sorted(list(set(rf2_mp_c30_main_exclsv+rf2_mf_c30_main_exclsv)))
	#print(rf2_main_exclusive)
	### FIND 57 EXCLUSIVE FEATURES ###

	rf2_pass_exclusive = sorted(list(set(rf2_mp_c30_pass_exclsv+rf2_pf_c30_pass_exclsv)))
	#print(rf2_pass_exclusive)
	### FIND 46 EXCLUSIVE FEATURES ###
	
	rf2_fail_exclusive = sorted(list(set(rf2_mf_c30_fail_exclsv+rf2_pf_c30_fail_exclsv)))
	#print(rf2_pass_exclusive)
	### FIND 53 EXCLUSIVE FEATURES ###

	rf2_all_exclusive = sorted(list(set(rf2_main_exclusive + rf2_pass_exclusive + rf2_fail_exclusive)))
	#print(rf2_pass_exclusive)
	## FIND 103 Exclusive Features ###


	# 피쳐뽑기
	rf2_c30_exc20_main = rf2_main_df.loc[:, rf2_main_exclusive]
	rf2_c30_exc20_pass = rf2_main_df.loc[:, rf2_pass_exclusive]
	rf2_c30_exc20_fail = rf2_main_df.loc[:, rf2_fail_exclusive]
	rf2_c30_exc20_all = rf2_main_df.loc[:, rf2_all_exclusive]

	# 37개의 Feature
	csv_file = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rf2_c30_exc20_main.items()]))
	csv_file.to_csv('./rf2_c30_exc20_main.csv')

	rf2_filt30_pass_df = rf2_main_df.loc[:, rf2_pass_exclusive]
	# 37개의 Feature
	csv_file = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rf2_c30_exc20_pass.items()]))
	csv_file.to_csv('./rf2_c30_exc20_pass.csv')

	rf2_filt30_fail_df = rf2_main_df.loc[:, rf2_fail_exclusive]
	# 33개의 Feature
	csv_file = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rf2_c30_exc20_fail.items()]))
	csv_file.to_csv('./rf2_c30_exc20_fail.csv')

	rf2_filt_all_df = rf2_main_df.loc[:, rf2_all_exclusive]
	# 69개의 Feature
	csv_file = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rf2_c30_exc20_all.items()]))
	csv_file.to_csv('./rf2_c30_exc20_sumset.csv')
	
	
	####################################################################################
	##  여기까지 데이터 시각화에 있어 선제했던 데이터 전처리. 시각화 데이터는 0525로 첨부  ##
	####################################################################################
	'''
	
	rf2_c30_main_df = correlation_refine(rf2_main_df, 0.30, 0.8)
	rf2_c30_pass_df = correlation_refine(rf2_pass_df, 0.30, 0.8)
	rf2_c30_fail_df = correlation_refine(rf2_fail_df, 0.30, 0.8)

	print("=============== STEP 2 ==============")
	print("\nmain : ")
	print(rf2_c30_main_df.describe())
	print("\n\npass : ")
	print(rf2_c30_pass_df.describe())
	print("\n\nfail : ")
	print(rf2_c30_fail_df.describe())
	print("\n\n")


	# 프로젝트 편의를 위해 데이터 저장
	rf2_c30_main_df.to_csv('./rf2_c30_main.csv')
	rf2_c30_pass_df.to_csv('./rf2_c30_pass.csv')
	rf2_c30_fail_df.to_csv('./rf2_c30_fail.csv')

	### 계산된 파일 불러오기
	rf2_f30_main = pd.read_csv('./rf2_c30_exc20_main.csv')
	rf2_f30_pass = pd.read_csv('./rf2_c30_exc20_pass.csv')
	rf2_f30_fail = pd.read_csv('./rf2_c30_exc20_fail.csv')
	rf2_f30_all = pd.read_csv('./rf2_c30_exc20_sumset.csv')

	# 퍼센트에 따른 결측치 상위 Feature 제거
	# 이미 파일이 존재하면 수행하지 않음
	if(os.path.isfile('./rf3_c30_exc30_m30_main')):
		rf3_f30_main = pd.read_csv('./rf3_c30_exc30_m30_main.csv')
		rf3_f30_pass = pd.read_csv('./rf3_c30_exc30_m30_pass.csv')
		rf3_f30_fail = pd.read_csv('./rf3_c30_exc30_m30_fail.csv')
		rf3_f30_all = pd.read_csv('./rf3_c30_exc30_m30_sumset.csv')
	else:
		# 결측치 처리 (STEP2 데이터의 피쳐들이 매우 적어 결측치 필터 기준을 높였습니다)
		rf3_f30_m30_main = missing_value_refine("./rf3_c30_exc30_m30_main.csv", rf2_f30_main, 0.3)
		rf3_f30_m30_pass = missing_value_refine('./rf3_c30_exc30_m30_pass.csv', rf2_f30_pass, 0.3)
		rf3_f30_m30_fail = missing_value_refine('./rf3_c30_exc30_m30_fail.csv', rf2_f30_fail, 0.3)
		rf3_f30_m30_all = missing_value_refine('./rf3_c30_exc30_m30_sumset.csv', rf2_f30_all, 0.3)
	
	# 결측치 제거와 보정
	# 이미 파일이 존재하면 수행하지 않음
	## 40%, 60% 의 차이를 구별하기 위해 시각화 필요
	if(os.path.isfile('./rf3_c30_m40_main.csv')):
		rf3_c30_m40_main = pd.read_csv('./rf3_c30_m40_main.csv')
		rf3_c30_m40_pass = pd.read_csv('./rf3_c30_m40_pass.csv')
		rf3_c30_m40_fail = pd.read_csv('./rf3_c30_m40_fail.csv')
	else:
		rf3_c30_m40_main = missing_value_refine("./rf3_c30_m40_main.csv", rf2_c30_main_df, 0.4)
		rf3_c30_m40_pass = missing_value_refine("./rf3_c30_m40_pass.csv", rf2_c30_pass_df, 0.4)
		rf3_c30_m40_fail = missing_value_refine("./rf3_c30_m40_fail.csv", rf2_c30_fail_df, 0.4)

	if(os.path.isfile('./rf3_c30_m40_main.csv')):
		rf3_c30_m40_main = pd.read_csv('./rf3_c30_m40_main.csv')
		rf3_c30_m40_pass = pd.read_csv('./rf3_c30_m40_pass.csv')
		rf3_c30_m40_fail = pd.read_csv('./rf3_c30_m40_fail.csv')
	else:
		rf3_c30_m40_main = missing_value_refine('./rf3_c30_m40_main.csv', rf2_c30_main_df, 0.3)
		rf3_c30_m40_pass = missing_value_refine('./rf3_c30_m40_pass.csv', rf2_c30_pass_df, 0.3)
		rf3_c30_m40_fail = missing_value_refine('./rf3_c30_m40_fail.csv', rf2_c30_fail_df, 0.3)

	if(os.path.isfile('./rf3_c30_m60_main.csv')):
		rf3_c30_m60_main = pd.read_csv('./rf3_c30_m60_main.csv')
		rf3_c30_m60_pass = pd.read_csv('./rf3_c30_m60_pass.csv')
		rf3_c30_m60_fail = pd.read_csv('./rf3_c30_m60_fail.csv')
	else:
		rf3_c30_m60_main = missing_value_refine("./rf3_c30_m60_main.csv", rf2_c30_main_df, 0.6)
		rf3_c30_m60_pass = missing_value_refine("./rf3_c30_m60_pass.csv", rf2_c30_pass_df, 0.6)
		rf3_c30_m60_fail = missing_value_refine("./rf3_c30_m60_fail.csv", rf2_c30_fail_df, 0.6)

	print("=============== STEP 3 ==============")
	print(rf3_c30_m40_main.describe())
	print(rf3_c30_m40_pass.describe())
	print(rf3_c30_m40_fail.describe())
	print()
	print(rf3_c30_m60_main.describe())
	print(rf3_c30_m60_pass.describe())
	print(rf3_c30_m60_fail.describe())
	return;
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






