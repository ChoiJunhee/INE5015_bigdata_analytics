#####################################################################
#####################################################################
#### SECOM Data Preprocessing and Data Modeling - Version 0.1.5 ####
#####################################################################
#####################################################################

############################# Update 0.1.5 ##########################
## 1. 데이터 전처리 과정에서 보다 더 효율적인 구조를 적용 하였습니다.   ##
## 2. 스텝을 직관적이고, 일반적인 정제 순서를 고려하여 적용 하였습니다. ##
#####################################################################

#####################################################################
#### 데이터 클리닝 과정 일부 수정
#### 1. Missing Value remove or Imputation.
#### 2. Ouliers remove or smoothing. 
#### 3. Oulier Remove -> Missing Value remove -> Missing Value Impute
#### 시각화, 기술적 통계를 통해 의미 없는 데이터셋 삭제 (코드)
##################################################################### 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

from sklearn.feature_selection import RFE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve

# @param 'Time'이 제거된 Pass/Fail Label 데이터프레임
# @result Confuse Matrix에 의한 결과 출력...
# @return (미구현) 점수 리스트 
def Confuse_Matrix_Performance(df):
	#https://injo.tistory.com/13
	#http://blog.naver.com/PostView.nhn?blogId=siniphia&logNo=221396370872
	X = df.iloc[1:, 1:]
	Y = df.iloc[1:, 0]
	print(" X ")
	print(X)
	print("\n Y ")
	print(Y)
	print()
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=777, test_size=0.3, stratify=Y)
	lr_clf = LogisticRegression()
	lr_clf.fit(X_train, Y_train)
	pred = lr_clf.predict(X_test)
	pred_proba = lr_clf.predict_proba(X_test)[:, 1:]

	matrix = confusion_matrix(Y_test, y_pred=pred)
	accuracy = accuracy_score(Y_test, pred)
	precision = precision_score(Y_test, pred)
	recall = recall_score(Y_test, pred)
	f1 = f1_score(Y_test, pred)
	roc_auc = roc_auc_score(Y_test, pred_proba)
	print("Confusion Matrix")
	print(matrix)
	print('Accuracy: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, F1-Score: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))


'''
@param - steps : 기 진행 되어 데이터 셋을 확보한 경우, 해당 단계부터 할 수 있도록 함
@result : 전처리 완료된 데이터 (csv file)
'''
def DataAnalytics(step):

	#반복문과 조건문을 통해 switch 역할을 하도록 함 */
	while(step != 10):

		if(step == 0):
			# raw 데이터를 정렬해서 가져옴
			df = raw_csv('./uci-secom.csv')

			#pre-std-remove (step 0 폴더에 1, 3, 5 셋 있습니다.)
			df = data_std_remove(df, 1);
			#print(df.describe())
			#sns.boxplot(data = df)
			#plt.show()
			df.to_csv('./[step 0] - rawfile_low_refine/[1]_std_1.0.csv', index=False)

			print("[*] Step 0 - Complete.")
			step = 1

		elif(step == 1):
			# 표준편차가 극히 적은 일부 피쳐 삭제된 데이터를
			# Pass / Fail / All 으로 나누는 과정
			df1 = pd.read_csv('./[step 0] - rawfile_low_refine/[1]_std_1.0.csv')
			df3 = pd.read_csv('./[step 0] - rawfile_low_refine/[2]_std_3.0.csv')

			all_df1, pass_df1, fail_df1 = devide_PF(df)
			all_df1.to_csv('./[step 1] - devide_PF/std10_all.csv', index=False)
			pass_df1.to_csv('./[step 1] - devide_PF/std10_pass.csv', index=False)
			fail_df1.to_csv('./[step 1] - devide_PF/std10_fail.csv', index=False)
			
			all_df3, pass_df3, fail_df3 = devide_PF(df)
			all_df1.to_csv('./[step 1] - devide_PF/std30_all.csv', index=False)
			pass_df1.to_csv('./[step 1] - devide_PF/std30_pass.csv', index=False)
			fail_df1.to_csv('./[step 1] - devide_PF/std30_fail.csv', index=False)
			print("[*] Step 1 - Complete.")
			step = 2

		elif(step == 2):
			# 기술적 통계 확인 : Pass 1463, Fail 104 으로 전체 데이터 데비 Fail은 6.6%
			# 데이터 시각화 확인 -> STD30 사용

			all_df = pd.read_csv('./[step 1] - devide_PF/std30_all.csv')
			pass_df = pd.read_csv('./[step 1] - devide_PF/std30_pass.csv')
			fail_df = pd.read_csv('./[step 1] - devide_PF/std30_fail.csv')

			#sns.boxplot(data=fail_df)
			#plt.show()

			# 시각화 처리 자동화 하고싶은데 시간이 너무 없음 ㅜㅜㅜ
			## 이 부분에서도 아직 많은 피쳐가 남아 다음 스탭 이후 다시 할 예정

			all_df, pass_df, fail_df = devide_PF(df)
			all_df.to_csv('./[step 2] - pass_fail_analysis/all.csv', index=False)
			pass_df.to_csv('./[step 2] - pass_fail_analysis/pass.csv', index=False)
			fail_df.to_csv('./[step 2] - pass_fail_analysis/fail.csv', index=False)
			print("[*] Step 2 - Complete.")

			step = 3
		elif(step == 3):
			# Feature간 상관관계 확인 및 제거 # STD 30 선정

			#std10_all = pd.read_csv('./[step 1] - devide_PF/std10_all.csv')
			#std10_pass = pd.read_csv('./[step 1] - devide_PF/std10_pass.csv')
			#std10_fail = pd.read_csv('./[step 1] - devide_PF/std10_fail.csv')

			std30_all = pd.read_csv('./[step 1] - devide_PF/std30_all.csv')
			std30_pass = pd.read_csv('./[step 1] - devide_PF/std30_pass.csv')
			std30_fail = pd.read_csv('./[step 1] - devide_PF/std30_fail.csv')


			## corr 0.3 / 0.6 큰 의미 없음 / std3 진행
			#s1_c30_all = correlation_remove(std10_all, 0.3, 0.8)
			#s1_c30_pass = correlation_remove(std10_pass, 0.3, 0.8)
			#s1_c30_fail = correlation_remove(std10_fail, 0.3, 0.8)

			s3_c30_all = correlation_remove(std30_all, 0.3, 0.8)
			s3_c30_pass = correlation_remove(std30_pass, 0.3, 0.8)
			s3_c30_fail = correlation_remove(std30_fail, 0.3, 0.8)

			#s1_c30_all.to_csv('./[step 3] - correlation/s1_c30_all.csv', index=False)
			#s1_c30_pass.to_csv('./[step 3] - correlation/s1_c30_pass.csv', index=False)
			#s1_c30_fail.to_csv('./[step 3] - correlation/s1_c30_fail.csv', index=False)

			s3_c30_all.to_csv('./[step 3] - correlation/s3_c30_all.csv', index=False)
			s3_c30_pass.to_csv('./[step 3] - correlation/s3_c30_pass.csv', index=False)
			s3_c30_fail.to_csv('./[step 3] - correlation/s3_c30_fail.csv', index=False)

			## 임시 시각화 (폴더에있음)
			#sns.boxplot(data=s3_c30_pass)
			#plt.show()

			#sns.boxplot(data=s3_c30_fail)
			#plt.show()

			#print(s3_c30_all) 

			print("[*] Step 3 - Complete.")
			print("[*] Step 3 - 202 Features")

			step = 4

		elif(step == 4):
			
			# 결측치 보정 단계
			s3_c30_all = pd.read_csv('./[step 3] - correlation/s3_c30_all.csv')
			s3_c30_pass = pd.read_csv('./[step 3] - correlation/s3_c30_pass.csv')
			s3_c30_fail = pd.read_csv('./[step 3] - correlation/s3_c30_fail.csv')

			## Missing Value Percentage > 0.45 : 삭제
			## 0.3 < MVP < 0.45 : 중앙 값 대체 
			## Missing Value Percentage < 0.3 : IterativeImputation

			s4_MVP_all = missing_value_processing(s3_c30_all)
			s4_MVP_pass = missing_value_processing(s3_c30_pass)
			s4_MVP_fail = missing_value_processing(s3_c30_fail)

			return;
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
			# 아직 명확하게 잘라내기 애매해서 패

			step = 6
			pass

		elif(step == 6):
			#스케일링 단계
			
			################################################################
			## step 5 확정이 되지 않아서 여기서 부터는 데이터셋이 늘어납니다. ##
			################################################################

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

			print("ALL | PASS | FAIL\n")
			for x in data_set1:
				print("MMS")
				Confuse_Matrix_Performance(x)
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
def missing_value_processing(df):
	## MVP > 0.45 : 삭제
	## 0.45 > MVP > 0.3 : 중앙값
	## 0.3 > MVP : Iterative Imputation (max_iter=20)
	null_data = df.isnull().sum()
	null_df = pd.DataFrame(null_data)
	null_df['null_per'] = (null_df[0] / len(null_df.index))

	null_list = null_df[null_df['null_per'] > 0.45].index
	remove_data = df.drop(null_list, axis=1)
	print(remove_data)
	return;
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
DataAnalytics(3)

############################ To be Updated ##########################
## 1. 미정                                                          ##
## 2. 미정                                                          ##
#####################################################################



