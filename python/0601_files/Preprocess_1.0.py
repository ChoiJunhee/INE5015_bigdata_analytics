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
from sklearn.ensemble import RandomForestClssifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


'''
@param - steps : 기 진행 되어 데이터 셋을 확보한 경우, 
                 해당 단계부터 할 수 있도록 함
@result : 전처리 완료된 데이터 (csv file)
'''
def DataAnalytics(step):

	#반복문과 조건문을 통해 switch 역할을 하도록 함 */
	while(step != 10):

		if(step == 0):
			# raw 데이터를 정렬해서 가져옴
			df = raw_csv('./uci-secom.csv')
			df.to_csv('./step0 - raw/secom.csv', index=False)
			print("[*] Step 0 - Complete.")
			step = 1

		elif(step == 1):
			step1_df = pd.read_csv('./step0 - raw/secom.csv')
			# 일정 표준편차 이하 제거 후 csv 저장
			step1_s5 = data_std_remove(step1_df, 0.5);
			step1_s5.to_csv('./step1 - std remove/std_0.5.csv', index=False)
			step1_s10 = data_std_remove(step1_df, 1);
			step1_s10.to_csv('./step1 - std remove/std_1.0.csv', index=False)
			print("[*] Step 1 - Complete.")
			step = 2

		elif(step == 2):
			# Pass Fail 데이터 나누는 단계
			step2_s5_df =pd.read_csv('./step1 - std remove/std_0.5.csv');
			step2_s10_df = pd.read_csv('./step1 - std remove/std_1.0.csv');

			# 실제 데이터셋 사용 시 s10 사용함
			#step2_all_df, step2_pass_df, step2_fail_df = devide_PF(step2_s5_df)
			#step2_all_df.to_csv('./step2 - devide PF/s5_all.csv', index=False)
			#step2_pass_df.to_csv('./step2 - devide PF/s5_pass.csv', index=False)
			#step2_fail_df.to_csv('./step2 - devide PF/s5_fail.csv', index=False)

			tep2_all_df, step2_pass_df, step2_fail_df = devide_PF(step2_s10_df)
			step2_all_df.to_csv('./step2 - devide PF/s10_all.csv', index=False)
			step2_pass_df.to_csv('./step2 - devide PF/s10_pass.csv', index=False)
			step2_fail_df.to_csv('./step2 - devide PF/s10_fail.csv', index=False)
			print("[*] Step 2 - Complete.")
			step = 3

		elif(step == 3):
			# Feature간 상관관계 확인 및 제거 #
			step3_s10_all = pd.read_csv('./step2 - devide PF/s10_all.csv')
			step3_s10_pass = pd.read_csv('./step2 - devide PF/s10_pass.csv')
			step3_s10_fail = pd.read_csv('./step2 - devide PF/s10_fail.csv')

			#corr 30만 사용함. corr 60과 큰 차이 없음. (데이터가 작은 Fail 제외)
			#fail 데이터에서 중요한 Feature가 있을 것으로 추측됨.
			step3_s10_c30_all = correlation_remove(step3_s10_all, 0.3, 0.8)
			step3_s10_c30_pass = correlation_remove(step3_s10_pass, 0.3, 0.8)
			step3_s10_c30_fail = correlation_remove(step3_s10_fail, 0.3, 0.8)

			step3_s10_c30_all.to_csv('./step3 - correlation/c30_all.csv', index=False)
			step3_s10_c30_pass.to_csv('./step3 - correlation/c30_pass.csv', index=False)
			step3_s10_c30_fail.to_csv('./step3 - correlation/c30_fail.csv', index=False)

			print("[*] Step 3 - Complete.")
			step = 4

		elif(step == 4):
			# 결측치 보정 단계
			step3_s10_c30_all = pd.read_csv('./step3 - correlation/c30_all.csv')
			step3_s10_c30_pass = pd.read_csv('./step3 - correlation/c30_pass.csv')
			step3_s10_c30_fail =pd.read_csv('./step3 - correlation/c30_fail.csv')
			
			# ALL, PASS, FAIL마다 결측치 커트라인을 변경함
			'''
			step4_m5_all = missing_value_processing(step3_s10_c30_all, 0.5)
			step4_m5_pass = missing_value_processing(step3_s10_c30_pass, 0.4)
			step4_m5_fail = missing_value_processing(step3_s10_c30_fail, 0.3)

			step4_m5_all.to_csv('./step4 - missing value/step4_m5_all.csv', index=False)
			step4_m5_pass.to_csv('./step4 - missing value/step4_m5_pass.csv', index=False)
			step4_m5_fail.to_csv('./step4 - missing value/step4_m5_fail.csv', index=False)
			'''
			step4_m3_all = missing_value_processing(step3_s10_c30_all, 0.3)
			step4_m3_pass = missing_value_processing(step3_s10_c30_pass, 0.2)
			step4_m3_fail = missing_value_processing(step3_s10_c30_fail, 0.1)

			step4_m3_all.to_csv('./step4 - missing value/step4_m3_all.csv', index=False)
			step4_m3_pass.to_csv('./step4 - missing value/step4_m3_pass.csv', index=False)
			step4_m3_fail.to_csv('./step4 - missing value/step4_m3_fail.csv', index=False)
			
			print("[*] Step 4 - Complete.")
			step = 5
			

		elif(step == 5):
			#이상치 보정 단계	
			step4_m3_all = pd.read_csv('./step4 - missing value/step4_m3_all.csv')
			step4_m3_pass = pd.read_csv('./step4 - missing value/step4_m3_pass.csv')
			step4_m3_fail = pd.read_csv('./step4 - missing value/step4_m3_fail.csv')
			newdf=set_data_scale(step4_m3_pass, 0)
			# 아직 명확하게 잘라내기 애매해서 패스
			step = 6
			pass

		elif(step == 6):
			#스케일링 단계
			## step 5 확정이 되지 않아서 여기서 부터는 데이터셋이 늘어납니다.
			step4_m3_all = pd.read_csv('./step4 - missing value/step4_m3_all.csv')
			step4_m3_pass = pd.read_csv('./step4 - missing value/step4_m3_pass.csv')
			step4_m3_fail = pd.read_csv('./step4 - missing value/step4_m3_fail.csv')

			a1, a2, a3, a4 = set_data_scale(step4_m3_all, 0)
			a1.to_csv('./step6 - data scale/step6_all_mms.csv', index=False)
			a2.to_csv('./step6 - data scale/step6_all_ss.csv', index=False)
			a3.to_csv('./step6 - data scale/step6_all_rs.csv', index=False)
			a4.to_csv('./step6 - data scale/step6_all_ns.csv', index=False)

			p1, p2, p3, p4 = set_data_scale(step4_m3_pass, 0)
			p1.to_csv('./step6 - data scale/step6_pass_mms.csv', index=False)
			p2.to_csv('./step6 - data scale/step6_pass_ss.csv', index=False)
			p3.to_csv('./step6 - data scale/step6_pass_rs.csv', index=False)
			p4.to_csv('./step6 - data scale/step6_pass_ns.csv', index=False)

			f1, f2, f3, f4 = set_data_scale(step4_m3_fail, 0)
			f1.to_csv('./step6 - data scale/step6_fail_mms.csv', index=False)
			f2.to_csv('./step6 - data scale/step6_fail_ss.csv', index=False)
			f3.to_csv('./step6 - data scale/step6_fail_rs.csv', index=False)
			f4.to_csv('./step6 - data scale/step6_fail_ns.csv', index=False)
			# 이 파일들을 이용해 Feature Selection을 진행합니다.

			step = 7
			
		elif(step == 7):
			#Feature Selection
			a1 = pd.read_csv('./step6 - data scale/step6_all_mms.csv')
			a2 = pd.read_csv('./step6 - data scale/step6_all_ss.csv')
			a3 = pd.read_csv('./step6 - data scale/step6_all_rs.csv')
			a4 = pd.read_csv('./step6 - data scale/step6_all_ns.csv')

			p1 = pd.read_csv('./step6 - data scale/step6_pass_mms.csv')
			p2 = pd.read_csv('./step6 - data scale/step6_pass_ss.csv')
			p3 = pd.read_csv('./step6 - data scale/step6_pass_rs.csv')
			p4 = pd.read_csv('./step6 - data scale/step6_pass_ns.csv')

			f1 = pd.read_csv('./step6 - data scale/step6_fail_mms.csv')
			f2 = pd.read_csv('./step6 - data scale/step6_fail_ss.csv')
			f3 = pd.read_csv('./step6 - data scale/step6_fail_rs.csv')
			f4 = pd.read_csv('./step6 - data scale/step6_fail_ns.csv')

			data_set1 = [a1, p1, f1]
			data_set2 = [a2, p2, f2]
			data_set3 = [a3, p3, f3]
			data_set4 = [a4, p4, f4]

			step = 8
			pass
			
		elif(step == 8):
			#오버 샘플링 단계

			step = 9
			pass

		elif(step == 9):
			#최종 데이터 셋 선정 단계

			step = 10
			print("[*] Preprocessing Complete.")
		else:
			pass
	# 반복문 종료 - Preprocessing Complete.


'''
@param - file_link : Raw CSV 파일의 주소
@return - 문자 열이 0, 1번째로 정렬된  CSV 파일의 Dataframe
'''
def raw_csv(file_link):
	raw_df = pd.read_csv(file_link)

	char_df = raw_df.loc[:, ['Time', 'Pass/Fail']]
	inte_df = raw_df.drop(['Time', 'Pass/Fail'], axis=1).add_prefix('F')
	return pd.concat([char_df, inte_df], axis=1)



'''
@param - df : 가공할 데이터 프레임
@param - num : 기준이 될 표준편차
@return - 기준 이하의 표준편차를 가진 Feature가 제거된 데이터 프레임
'''
def data_std_remove(df, num):
	df_trans = df.describe().transpose()
	remove_std = df_trans[df_trans['std'] <= num].index
	result = df.drop(remove_std, axis=1)
	return df.drop(remove_std[1:], axis=1)

'''
@param - df : PASS/FAIL을 나눌 프레임
@return - Pass, Fail로 나눈 데이터 프레임
'''
def devide_PF(df):
	pass_df = df[df['Pass/Fail'] == -1]
	fail_df = df[df['Pass/Fail'] == 1]
	return df, pass_df, fail_df


'''
@param - df : 다중공선성을 제거할 데이터 프레임
@param - num : 기준이 될 상관관계 지수
@return - 상관관계가 높은 Feature가 제거된 데이터 프레임
'''
def correlation_remove(df, per, abs_num):
	corr_df = df.corr()
	corr_nan_df = corr_df[corr_df > abs(abs_num)]
	cols = list(corr_nan_df)
	rows = list(corr_nan_df.index)

	corr_list = []

	# 특정 계수 이상의 상관관계를 가진 Feature를 찾는 중 ...
	for i in range(0, len(cols)):
		for j in range(0, len(rows)):
			temp = []
			if(corr_nan_df[cols[i]][rows[j]] > abs(abs_num)):
				temp.append(cols[i])
				temp.append(rows[j])
				temp.append(corr_nan_df[cols[i]][rows[j]])
				corr_list.append(temp)

	
	corr_list_df = pd.DataFrame(corr_list)
	corr_result = corr_list_df.drop_duplicates([2], keep="first")

	x = corr_result[0].value_counts()
	xdf = pd.DataFrame(x)
	y = corr_result[1].value_counts()
	ydf = pd.DataFrame(y)

	
	corr_df = pd.concat([xdf, ydf], ignore_index=True, axis=1)
	corr_pc = corr_df.fillna(0)
	corr_df['sum'] = corr_pc[0] + corr_pc[1]
	corr_df = corr_df.sort_values(by=['sum'], axis=0, ascending=False)

	extract = []
	for i in range(0, int(len(corr_df.index) * per)):
		extract.append(list(corr_df.index)[i])

	return df.drop(extract, axis=1)


'''
@param - df : 결측치를 제거할 데이터 프레임
@param - per : 기준이 될 결측치 비율
@return - 기준 이상의 결측치를 갖는 Feature가 삭제된 데이터 프레임
'''
def missing_value_processing(df, per):

	null_data = df.isnull().sum()
	null_df = pd.DataFrame(null_data)
	null_df['null_per'] = (null_df[0] / len(null_df.index))

	null_list = null_df[null_df['null_per'] > per].index
	remove_data = df.drop(null_list, axis=1)
	save_cols = list(remove_data.describe().columns)

	try:
		save_cols = list(remove_data.describe().columns)
		save_char_df = remove_data.loc[:, ['Time', 'Pass/Fail']]
		imp_data = remove_data.drop(['Time', "Pass/Fail"], axis=1)
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=16, verbose=False).fit_transform(imp_data), columns=save_cols[1:])
		processed_df = pd.concat([save_char_df, imputed_df], axis=1)
	except:
		save_cols = list(remove_data.describe().columns)
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=16, verbose=False).fit_transform(remove_data), columns=save_cols)
		processed_df = imputed_df
	return processed_df


def outlier_processing(df, weight):
	df_save = df.loc[:, ['Time', 'Pass/Fail']]
	df = df.drop(['Time', 'Pass/Fail'], axis=1)
	cell_num = 0
	over_list = []
	for i in range(0, len(list(df.columns))):
		df_rows = df[list(df.columns)[i]]
		quant25 = np.percentile(df_rows.values, 2.5)
		quant75 = np.percentile(df_rows.values, 97.5)

		iqr = quant75 - quant25
		iqr_weight = iqr * weight

		min_q = quant25 - iqr_weight
		max_q = quant75 + iqr_weight

		outlier_index = list(df_rows[(df_rows < min_q) | (df_rows > max_q)].index)
		minimum_index = list(df_rows[(df_rows < min_q)].index)
		maximum_index = list(df_rows[(df_rows > max_q)].index)

		if(len(outlier_index) != 0):
			cell_num += len(outlier_index)

			if(len(outlier_index) > len(df.columns) * 0.1):
				over_list.append(list(df.columns)[i])
				over_list.append(len(outlier_index))

		for idx in range(0, len(minimum_index)):
			df.loc[[minimum_index[idx]], [list(df.columns)[i]]] = min_q
		for idx in range(0, len(maximum_index)):
			df.loc[[maximum_index[idx]], [list(df.columns)[i]]] = max_q

def set_data_scale(df, num):
	redf = df.iloc[:,1:]
	MMS = MinMaxScaler()
	SS = StandardScaler()
	RS = RobustScaler()
	NS = Normalizer()

	MMS.fit(redf)
	SS.fit(redf)
	RS.fit(redf)
	NS.fit(redf)

	redf_mms = MMS.transform(redf)
	redf_ss = SS.transform(redf)
	redf_rs = RS.transform(redf)
	redf_ns = NS.transform(redf)

	redf_mms_pd = pd.DataFrame(redf_mms, columns=redf.columns)
	redf_ss_pd = pd.DataFrame(redf_ss, columns=redf.columns)
	redf_rs_pd = pd.DataFrame(redf_rs, columns=redf.columns)
	redf_ns_pd = pd.DataFrame(redf_ns, columns=redf.columns)

	return redf_mms_pd, redf_ss_pd, redf_rs_pd, redf_ns_pd

def feature_selection(df, num):
	pass

def data_oversampling(df, num):
	pass





# @param : 시작하고 싶은 전처리 단계
DataAnalytics(5)

############################ To be Updated ##########################
## 1. 미정                                                          ##
## 2. 미정                                                          ##
#####################################################################