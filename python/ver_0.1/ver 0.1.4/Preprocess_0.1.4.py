#####################################################################
#####################################################################
#### SECOM Data Preprocessing and Data Modeling - Version 0.1.4 ####
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
			# Pass / Fail / All 으로 나누는 과정 -> std 3.0
			#df1 = pd.read_csv('./[step 0] - rawfile_low_refine/[1]_std_1.0.csv')
			df3 = pd.read_csv('./[step 0] - rawfile_low_refine/[2]_std_3.0.csv')

			#all_df1, pass_df1, fail_df1 = devide_PF(df)
			#all_df1.to_csv('./[step 1] - devide_PF/std10_all.csv', index=False)
			#pass_df1.to_csv('./[step 1] - devide_PF/std10_pass.csv', index=False)
			#fail_df1.to_csv('./[step 1] - devide_PF/std10_fail.csv', index=False)
			
			all_df3, pass_df3, fail_df3 = devide_PF(df)
			all_df3.to_csv('./[step 1] - devide_PF/std30_all.csv', index=False)
			pass_df3.to_csv('./[step 1] - devide_PF/std30_pass.csv', index=False)
			fail_df3.to_csv('./[step 1] - devide_PF/std30_fail.csv', index=False)
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
			## -> 변경 0.6 이상 삭제, 이하 max_iter 30으로 진행. (비교필요)

			s4_MVP_all = missing_value_processing(s3_c30_all)
			s4_MVP_pass = missing_value_processing(s3_c30_pass)
			s4_MVP_fail = missing_value_processing(s3_c30_fail)

			
			s4_MVP_all.to_csv('./[step 4] - DC - Missing_Value_Inputation/m45_all.csv', index=False)
			s4_MVP_pass.to_csv('./[step 4] - DC - Missing_Value_Inputation/m45_pass.csv', index=False)
			s4_MVP_fail.to_csv('./[step 4] - DC - Missing_Value_Inputation/m45_fail.csv', index=False)

			print("[*] Step 4 - Complete.")
			step = 5
			

		elif(step == 5):
			#이상치 보정 단계	

			s4_all = pd.read_csv('./[step 4] - DC - Missing_Value_Inputation/m45_all.csv')
			s4_pass = pd.read_csv('./[step 4] - DC - Missing_Value_Inputation/m45_pass.csv')
			s4_fail = pd.read_csv('./[step 4] - DC - Missing_Value_Inputation/m45_fail.csv')
			
			s5_w15_p50_pass = outlier_processing(s4_pass, 1.5, 5)
			s5_w15_p50_fail = outlier_processing(s4_fail, 1.5, 6)
			s5_w15_p50_all = outlier_processing(s4_all, 1.5, 5.2)

			s4_all.to_csv('./[step 5] - DC - Oulier_refine/w15_o52_all.csv', index=False)
			s4_pass.to_csv('./[step 5] - DC - Oulier_refine/w15_o50_pass.csv', index=False)
			s4_fail.to_csv('./[step 5] - DC - Oulier_refine/w15_o60_fail.csv', index=False)
			
			print("[*] Step 5 - Complete.")
			step = 6

		elif(step == 6):
			# 스케일링 단계
			# ref : https://mizykk.tistory.com/101
			# 이상치 weight에 따라 조절이 가능하므로 표준화 스케일러, MINMAX 스케일러만 사용
			s4_all = pd.read_csv('./[step 5] - DC - Oulier_refine/w15_o52_all.csv')
			s4_pass = pd.read_csv('./[step 5] - DC - Oulier_refine/w15_o52_all.csv')
			s4_fail = pd.read_csv('./[step 5] - DC - Oulier_refine/w15_o52_all.csv')
			
			MINMAX_Scale_all, STD_Scale_all = set_data_scale(s4_all)
			MINMAX_Scale_pass, STD_Scale_pass = set_data_scale(s4_pass)
			MINMAX_Scale_fail, STD_Scale_fail = set_data_scale(s4_fail)

			MINMAX_Scale_all.to_csv('./[step 6] - DC - scaling/MINMAX_all.csv', index=False)
			STD_Scale_all.to_csv('./[step 6] - DC - scaling/STD_all.csv', index=False)
			MINMAX_Scale_pass.to_csv('./[step 6] - DC - scaling/MINMAX_pass.csv', index=False)
			STD_Scale_pass.to_csv('./[step 6] - DC - scaling/STD_pass.csv', index=False)
			MINMAX_Scale_fail.to_csv('./[step 6] - DC - scaling/MINMAX_fail.csv', index=False)
			STD_Scale_fail.to_csv('./[step 6] - DC - scaling/STD_fail.csv', index=False)
			print(MINMAX_Scale_all)
			print("[*] Step 6 - Complete.")

			step = 7
			return;
			
		elif(step == 7):
			#Feature Selection

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

	null_data = df.isnull().sum()
	null_df = pd.DataFrame(null_data)
	null_df['null_per'] = (null_df[0] / len(null_df.index))

	null_list = null_df[null_df['null_per'] > 0.6].index
	remove_data = df.drop(null_list, axis=1)
	save_cols = list(remove_data.describe().columns)

	try:
		save_cols = list(remove_data.describe().columns)
		save_char_df = remove_data.loc[:, ['Time', 'Pass/Fail']]
		imp_data = remove_data.drop(['Time', "Pass/Fail"], axis=1)
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=30, verbose=False).fit_transform(imp_data), columns=save_cols[1:])
		processed_df = pd.concat([save_char_df, imputed_df], axis=1)
	except:
		save_cols = list(remove_data.describe().columns)
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=30, verbose=False).fit_transform(remove_data), columns=save_cols)
		processed_df = imputed_df
	return processed_df

def outlier_processing(df, weight, percent):
	df_save = df.loc[:, ['Time', 'Pass/Fail']]
	df = df.drop(['Time', 'Pass/Fail'], axis=1)
	features = list(df.columns)
	remove_list = []
	outlier_over = []

	for feature in features:
		min_num = df[feature].min()
		max_num = df[feature].max()
		med_num = df[feature].median()

		q25 = df[feature].quantile(0.25)
		q50 = df[feature].quantile(0.5)
		q75 = df[feature].quantile(0.75)

		IQR = round(q75 - q25, 2)
		IQR_min = round(q25 - (weight * IQR),2)
		IQR_max = round(q75 + (weight * IQR),2)

		search = df[(df[feature] < (q25 - 1.5 * IQR)) | (df[feature] > (q75 + 1.5 * IQR))]
		
		per = round(100 * len(search)/len(df[feature]),2)

		print(feature + " - Percent : " + str(per))

		#print("MED : " + str(med_num))
		#print("IQR : " + str(IQR))
		#print("IQR_MIN : " + str(IQR_min))
		#print("IQR_MAX : " + str(IQR_max)+"\n")
		outlier_over.append(per)
		if(per >= percent):
			remove_list.append(str(feature))

	#print("Total Per : " + str(100*round(len(remove_list)/len(features), 2)))
	df = df.drop(remove_list, axis=1)
	return pd.concat([df_save, df], ignore_index=True, axis=1)


def set_data_scale(df):
	col_save = df.loc[:, ['Time', 'Pass/Fail']]
	df = df.drop(['Time', 'Pass/Fail'], axis=1)

	MINMAX = MinMaxScaler()
	Standard = StandardScaler()

	# 여기서 트레인셋 분리가 없으므로 트랜스폼만 진행
	MINMAX = pd.DataFrame(MINMAX.fit_transform(df))
	Standard = pd.DataFrame(Standard.fit_transform(df))

	MM = pd.concat([col_save, MINMAX], axis=1)
	SS = pd.concat([col_save, Standard], axis=1)

	return MM, SS



	return 

def feature_selection(df, num):
	pass

def data_oversampling(df, num):
	pass





# @param : 시작하고 싶은 전처리 단계
DataAnalytics(6)

############################ To be Updated ##########################
## 1. 미정                                                          ##
## 2. 미정                                                          ##
#####################################################################



