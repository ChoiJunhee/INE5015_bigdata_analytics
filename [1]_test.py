#####################################################################
#####################################################################
#### SECOM Data Preprocessing and Data Modeling - Version 0.2.3  ####
#####################################################################
#####################################################################

#####################################################################
##################################################################### 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import xgboost
import warnings

from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import CondensedNearestNeighbour, RandomUnderSampler

'''
	LogisticRegression
	0.9153606858988838
	{'max_iter': 175, 'penalty': 'none'}

	DecisionTreeClassifier
	0.8500169549759121
	{'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 0.01, 'min_samples_split': 6}

	RandomForestClassifier
	0.9902968584231788
	{'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}

	XGBClassifier
	0.9863600371906184
	{'colsample_bytree': 0.5, 'gamma': 0.15, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 0.1, 'subsample': 0.4}
'''

# @param 'Time'이 제거된 Pass/Fail Label 데이터프레임
# @result Confuse Matrix에 의한 결과 출력...
# @return 점수 리스트 = train_test_split(X, Y, test_size=0.2, stratify=Y)
def Confuse_Matrix_Performance(X_train, X_test, Y_train, Y_test, n):
	warnings.filterwarnings(action='ignore')

	if(n==0):
		clf = LogisticRegression(max_iter=200, penalty='l2')
	elif(n==1):
		clf = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=0.01, min_samples_split=6 )
	elif(n==2):
		clf = RandomForestClassifier(max_depth= 8, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=100)
	else:
		clf = xgboost.XGBClassifier(colsample_bytree=0.5, gamma=0.2, learning_rate=0.15, max_depth=8, n_estimators=100, reg_alpha= 0.05, reg_lambda=0.1, subsample=0.4)
	
	clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)
	roc_auc = roc_auc_score(Y_test, pred)

	matrix = confusion_matrix(Y_test, y_pred=pred)
	accuracy = accuracy_score(Y_test, pred)
	precision = precision_score(Y_test, pred)
	recall = recall_score(Y_test, pred)
	f1 = f1_score(Y_test, pred)
	
	return accuracy, precision, recall, f1, roc_auc

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
			df = data_std_remove(df, 0.5);
			
			df.to_csv('./[step 0] - rawfile_low_refine/[0]_std_0.5.csv', index=False)

			print("[*] Step 0 - Complete.")
			step = 1

		elif(step == 1):
			# 표준편차가 극히 적은 일부 피쳐 삭제된 데이터를
			# Pass / Fail / All 으로 나누는 과정 -> std 3.0
			df1 = pd.read_csv('./[step 0] - rawfile_low_refine/[0]_std_0.5.csv')
			#df3 (std 3.0 제거버전)은 사용하지 않습니다. df1 으로 확정되었습니다.

			all_df1, pass_df1, fail_df1 = devide_PF(df1)
			all_df1.to_csv('./[step 1] - devide_PF/std10_all.csv', index=False)
			pass_df1.to_csv('./[step 1] - devide_PF/std10_pass.csv', index=False)
			fail_df1.to_csv('./[step 1] - devide_PF/std10_fail.csv', index=False)
			print("[*] Step 1 - Complete.")
			step = 2

		elif(step == 2):
			# 기술적 통계 확인 : Pass 1463, Fail 104 으로 전체 데이터 데비 Fail은 6.6%
			# 데이터 시각화 확인 -> STD10 사용

			all_df = pd.read_csv('./[step 1] - devide_PF/std10_all.csv')
			pass_df = pd.read_csv('./[step 1] - devide_PF/std10_pass.csv')
			fail_df = pd.read_csv('./[step 1] - devide_PF/std10_fail.csv')

			'''
			시각화 테스트 공간 (STEP 1 -> STEP 3으로 진행되며, 이 단계는  테스트 단계)
			'''

			all_df, pass_df, fail_df = devide_PF(df)
			all_df.to_csv('./[step 2] - pass_fail_analysis/all.csv', index=False)
			pass_df.to_csv('./[step 2] - pass_fail_analysis/pass.csv', index=False)
			fail_df.to_csv('./[step 2] - pass_fail_analysis/fail.csv', index=False)
			print("[*] Step 2 - Complete.")

			step = 3
		elif(step == 3):
			# Feature간 상관관계 확인 및 제거 # STD10 진행 

			std30_all = pd.read_csv('./[step 1] - devide_PF/std10_all.csv')
			std30_pass = pd.read_csv('./[step 1] - devide_PF/std10_pass.csv')
			std30_fail = pd.read_csv('./[step 1] - devide_PF/std10_fail.csv')

			## corr 0.3 / 0.6 큰 의미 없음 / std3 진행
			s3_c30_all = correlation_remove(std30_all, 0.3, 0.8)
			s3_c30_pass = correlation_remove(std30_pass, 0.3, 0.8)
			s3_c30_fail = correlation_remove(std30_fail, 0.3, 0.8)

			s3_c30_all.to_csv('./[step 3] - correlation/s3_c30_all.csv', index=False)
			s3_c30_pass.to_csv('./[step 3] - correlation/s3_c30_pass.csv', index=False)
			s3_c30_fail.to_csv('./[step 3] - correlation/s3_c30_fail.csv', index=False)

			print("[*] Step 3 - Complete.")
			
			step = 4

		elif(step == 4):

			# 결측치 보정 단계
			s3_c30_all = pd.read_csv('./[step 3] - correlation/s3_c30_all.csv')
			s3_c30_pass = pd.read_csv('./[step 3] - correlation/s3_c30_pass.csv')
			s3_c30_fail = pd.read_csv('./[step 3] - correlation/s3_c30_fail.csv')

			#결측치 처리 -> 0.6 이상 비율인 경우 삭제
			#결측치 처리 -> 0.6 미만 비율인 경우 Imputation (max-iter=30)

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


			s5_w15_p50_pass = outlier_change(s4_pass, 1.5)
			s5_w15_p50_fail = outlier_change(s4_fail, 1.5)
			s5_w15_p50_all = outlier_change(s4_all, 1.5)

			s5_w15_p50_all = data_std_remove(s5_w15_p50_all, 0.5)
			s5_w15_p50_pass = data_std_remove(s5_w15_p50_pass, 0.5)
			s5_w15_p50_fail = data_std_remove(s5_w15_p50_fail, 0.5)
			
			# 이상치 처리 과정에서 마지막 per 부분 (피쳐 개수 고정시킬 용도) 삭제하고
			# IQR 방식 그대로 제거 (기존에는 일부 피쳐들이 생략되었음)

			s5_w15_p50_all.to_csv('./[step 5] - DC - Oulier_refine/w10_all.csv', index=False)
			s5_w15_p50_pass.to_csv('./[step 5] - DC - Oulier_refine/w10_pass.csv', index=False)
			s5_w15_p50_fail.to_csv('./[step 5] - DC - Oulier_refine/w10_fail.csv', index=False)
			
			print("[*] Step 5 - Complete.")
			step = 6

		elif(step == 6):
			# 스케일링 단계
			# 표준화 스케일러, MINMAX 스케일러만 사용
			s4_all = pd.read_csv('./[step 5] - DC - Oulier_refine/w10_all.csv')
			s4_pass = pd.read_csv('./[step 5] - DC - Oulier_refine/w10_pass.csv')
			s4_fail = pd.read_csv('./[step 5] - DC - Oulier_refine/w10_fail.csv')


			# MINMAX / STD 두 개로 나누어 진행
			MINMAX_Scale_all, STD_Scale_all = set_data_scale(s4_all)
			MINMAX_Scale_pass, STD_Scale_pass = set_data_scale(s4_pass)
			MINMAX_Scale_fail, STD_Scale_fail = set_data_scale(s4_fail)


			MINMAX_Scale_all.to_csv('./[step 6] - DC - scaling/MINMAX_all.csv', index=False)
			STD_Scale_all.to_csv('./[step 6] - DC - scaling/STD_all.csv', index=False)
			
			MINMAX_Scale_pass.to_csv('./[step 6] - DC - scaling/MINMAX_pass.csv', index=False)
			STD_Scale_pass.to_csv('./[step 6] - DC - scaling/STD_pass.csv', index=False)
			
			MINMAX_Scale_fail.to_csv('./[step 6] - DC - scaling/MINMAX_fail.csv', index=False)
			STD_Scale_fail.to_csv('./[step 6] - DC - scaling/STD_fail.csv', index=False)
			
			print("[*] Step 6 - Complete.")

			step = 7
			
		elif(step == 7):
			# Feature Selection
			## MINMAX와 STD간의 차이가 매우 적어 MMS으로 진행합니다.

			MINMAX_Scale_all = pd.read_csv('./[step 6] - DC - scaling/MINMAX_all.csv')
			#STD_Scale_all = pd.read_csv('./[step 6] - DC - scaling/STD_all.csv')
			
			MINMAX_Scale_pass = pd.read_csv('./[step 6] - DC - scaling/MINMAX_pass.csv')
			#STD_Scale_pass = pd.read_csv('./[step 6] - DC - scaling/STD_pass.csv')
			
			MINMAX_Scale_fail = pd.read_csv('./[step 6] - DC - scaling/MINMAX_fail.csv')
			#STD_Scale_fail = pd.read_csv('./[step 6] - DC - scaling/STD_fail.csv')
			
			KBS_MMS_ALL, RFE_MMS_ALL = feature_selection(MINMAX_Scale_all, 0.75)
			KBS_STD_ALL, RFE_STD_ALL = feature_selection(MINMAX_Scale_all, 0.75)

			#KBS_MMS_ALL.to_csv('./[step 7] Feature_Selection/[0]_KBS_MMS_All.csv', index=False)
			#KBS_STD_ALL.to_csv('./[step 7] Feature_Selection/[1]_KBS_STD_All.csv', index=False)


			#visual2(KBS_MMS_FAIL)
			#visual2(KBS_STD_FAIL)
			#visual2(RFE_MMS_FAIL)
			#visual2(RFE_STD_FAIL)

			#print("KBS")
			#Confuse_Matrix_Performance(KBS_MMS_ALL)
			#Confuse_Matrix_Performance(KBS_STD_ALL)
			#print("RFE")
			#Confuse_Matrix_Performance(RFE_MMS_ALL)
			#Confuse_Matrix_Performance(RFE_STD_ALL)

			## 확인 결과 MMS와 STD간의 큰 차이는 없었음 ##

			RFE_MMS_ALL.to_csv('./[step 7] Feature_Selection/[0]_RFE_MMS_All.csv', index=False)
			RFE_STD_ALL.to_csv('./[step 7] Feature_Selection/[1]_RFE_STD_All.csv', index=False)
			KBS_MMS_ALL.to_csv('./[step 7] Feature_Selection/[0]_KBS_MMS_All.csv', index=False)
			KBS_STD_ALL.to_csv('./[step 7] Feature_Selection/[1]_KBS_STD_All.csv', index=False)
			print("[*] Step 7 - Complete.")
			step = 8
			
		elif(step == 8):
			# oversampling, downsampling test
			FINAL_SET = pd.read_csv('./[step 7] Feature_Selection/[0]_RFE_MMS_All.csv')
			#RFE_STD_ALL = pd.read_csv('./[step 7] Feature_Selection/[1]_RFE_STD_All.csv')
			#KBS_MMS_ALL = pd.read_csv('./[step 7] Feature_Selection/[0]_KBS_MMS_All.csv')
			#KBS_STD_ALL = pd.read_csv('./[step 7] Feature_Selection/[1]_KBS_STD_All.csv')
			#RFE_MMS_TEST = correlation_remove(RFE_MMS_TEST, 0.1, 0.8)
			'''
			test = pd.read_csv('./[step 1] - devide_PF/std10_all.csv')
			test = test.drop(['Time', 'Pass/Fail'], axis=1).corr()
			mask = np.zeros_like(test, dtype=np.bool)
			mask[np.triu_indices_from(mask)] =True 
			sns.heatmap(test, cmap='Greens', annot=False, mask=mask, vmin=0, vmax=1)
			plt.show()
			'''

			X = FINAL_SET.iloc[:, 2:]
			Y = FINAL_SET.iloc[:, 1]

			data_performance(X, Y)
			
			### smote_LOGISTIC PERFORM_에서는 corr 제거가 부정적 영향을 끼침

			# Performance Verification [ FINAL ]
			
			print("[*] Preprocessing Complete.")
			break
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
	remove_std = df_trans[df_trans['std'] < num].index
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
@param - df 
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
	if("Pass/Fail" in extract):
		extract.remove("Pass/Fail")
	return df.drop(extract, axis=1)


'''
@param - df : 결측치를 제거할 데이터 프레임
@return - 기준 이상의 결측치를 갖는 Feature는 삭제, 이외엔 보정된 데이터 프레임
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


'''
!!! 버그로 인해 사용하지 않음 !!! 
@param - weight : IQR 가중치
@param - percent : 제거 할 때 특정 비율 이상 제거하도록 함. (세부 조정용)
'''
def outlier_processing(df, weight, percent):
	df_save = df.loc[:, ['Time', 'Pass/Fail']]
	df = df.drop(['Time', 'Pass/Fail'], axis=1)
	features = list(df.columns)
	remove_list = []
	outlier_over = []

	for feature in features:
		#min_num = df[feature].min()
		#max_num = df[feature].max()
		#med_num = df[feature].median()

		q25 = df[feature].quantile(0.25)
		q50 = df[feature].quantile(0.5)
		q75 = df[feature].quantile(0.75)

		IQR = round(q75 - q25, 2)
		IQR_min = round(q25 - (weight * IQR),2)
		IQR_max = round(q75 + (weight * IQR),2)

		search = df[(df[feature] < (q25 - 1.5 * IQR)) | (df[feature] > (q75 + 1.5 * IQR))]
		print(search)
		remove_list.append(search)
		
	df = df.drop(remove_list, axis=1)
	return pd.concat([df_save, df], axis=1)


'''
@param - df : dataframe
@return MINMAX, STD 스케일링을 거친 dataframe (2개)
'''
def set_data_scale(df):
	col_save = df.loc[:, ['Time', 'Pass/Fail']]
	save = df.columns
	df = df.drop(['Time', 'Pass/Fail'], axis=1)

	MINMAX = MinMaxScaler()
	Standard = StandardScaler()
	Normalize = Normalizer()

	# 여기서 트레인셋 분리가 없으므로 트랜스폼만 진행
	# Normalize 확인 결과 역시 필요가 없었음. MM, SS 로 진행

	MINMAX = pd.DataFrame(MINMAX.fit_transform(df))
	Standard = pd.DataFrame(Standard.fit_transform(df))

	MM = pd.concat([col_save, MINMAX], axis=1)
	SS = pd.concat([col_save, Standard], axis=1)
	MM.columns = save
	SS.columns = save
	return MM, SS

def feature_selection(df, per):
	# Wrapper는 실제 모델을 학습하면서 부분집합 중 중요한 것들을 얻어내는데
	# 이번 프로젝트에서는 모델 학습이 들어가는지 안들어가는지 몰라서
	# 일단 여러 방법 중 사이킷 런이 제공하는 몇 개의 방법으로 FE를 진행한다.
	# Filter Method를 통해 영향력이 있는 피쳐를 파악한 뒤
	# Wrapper Method를 사용하고 영향력이 있는 피쳐들을 집중 분석한다.

	#SelectKBest (Univariate Selection - 각 피쳐간의 통계적 관계)
	#Recursive Feature Elimination
	#Principal Component Analysis
	#Feature Importance check (RandomForest / ExtraTrees)

	# Reference : https://velog.io/@yeonha0422/Feature-Selection
	# Reference : https://www.kaggle.com/harangdev/feature-selection


	## 단일 변수 선택 방식
	df_cols = list(df.describe().columns)
	Y_pf = df['Pass/Fail'].values
	save = df.loc[:,['Time', 'Pass/Fail']]

	df = df.drop(['Time', 'Pass/Fail'], axis=1)

	select_df = SelectKBest(chi2, k=(int(per * len(df.columns)))).fit_transform(df, Y_pf)
	select_df = pd.DataFrame(select_df)
	KBS_ = pd.concat([save, select_df], axis=1)

	# 현재 피쳐 수 93(0.75기준) SelectKBest 의 내용은 레퍼런스 보몀 나와있음!
	## RFECV는 k-fold validation 이 필요한데, Fail의 개수가 너무 적어 사용 안함. 

	model = LogisticRegression(solver='liblinear')
	RFE_df = RFE(model, n_features_to_select=(int(per * len(df.columns))), step=1).fit(df, Y_pf)

	rank_df = pd.DataFrame(RFE_df.ranking_)
	rank_df.index = df.columns
	rank_list = list(RFE_df.ranking_)
	rank_df.columns = ['prior']
	drop_list = list(rank_df[rank_df['prior'] > 1].index)
	df = df.drop(drop_list, axis=1)
	RFE_ = pd.concat([save, df], axis=1)
	return KBS_, RFE_


'''
@kang961105 새 이상치 처리 알고리즘
'''
def outlier_change(df, weight):
    df_save = df.loc[:, ['Time', 'Pass/Fail']]
    df = df.drop(['Time', 'Pass/Fail'], axis=1)
    for i in range(0, len(list(df.columns))):
        a = df[list(df.columns)[i]]
        quantile_25 = np.percentile(a.values, 25)
        quantile_75 = np.percentile(a.values, 75)
        iqr = quantile_75 - quantile_25
        iqr_weight = iqr * weight
        lowest_val = quantile_25 - iqr_weight
        highest_val = quantile_75 + iqr_weight

        outlier_index = list(a[(a<lowest_val)|(a>highest_val)].index)
        lowest_index = list(a[(a<lowest_val)].index)
        highest_index = list(a[(a>highest_val)].index)

        if(len(outlier_index) != 0) : 
        	for index in range(0, len(lowest_index)):
        		df.loc[[lowest_index[index]],[list(df.columns)[i]]] = lowest_val

        	for  index in range(0, len(highest_index)):
        		df.loc[[ highest_index[index]],[list(df.columns)[i]]] =  highest_val
    
    return pd.concat([df_save, df], axis=1)


def data_performance(X, Y):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
	warnings.filterwarnings(action='ignore')
	
	Y_train = pd.DataFrame(Y_train, columns=['Pass/Fail'])
	NEWDF1 = pd.concat([Y_train, X_train], axis=1)
	NEWDF1 = data_oversampling(NEWDF1)
	Y_train = NEWDF1.iloc[:, 0]
	X_train = NEWDF1.iloc[:, 1:]

	Y_test = pd.DataFrame(Y_test, columns=['Pass/Fail'])
	NEWDF2 = pd.concat([Y_test, X_test], axis=1)
	NEWDF2 = data_oversampling(NEWDF2)
	Y_test = NEWDF2.iloc[:, 0]
	X_test = NEWDF2.iloc[:, 1:]


	#HyperParameter Tuning은  0.3 버전에서 추가 업데이트 예정입니다.
	'''
	LR_CLF = LogisticRegression()
	params={'penalty':['l1', 'l2', 'elasticnet', 'none'],
			'max_iter':[150, 175, 200, 225, 250, 275],
			}

	grid_clf = GridSearchCV(LR_CLF, param_grid=params, scoring='f1', cv=5)
	grid_clf.fit(X_train, Y_train)
	LR_PARAM = grid_clf.best_params_
	print("LogisticRegression")
	print(grid_clf.best_score_)
	print(grid_clf.best_params_)
	print()


	# DT_GCV
	DT_CLF = DecisionTreeClassifier()
	params = {'criterion':['gini', 'entropy'],
				  'max_depth':[1, 3, 5],
				  'min_samples_split':[4, 6, 8],
				  'min_samples_leaf':[0.01, 0.03, 0.05]
				 }
	grid_dt = GridSearchCV(DT_CLF, param_grid=params, scoring='f1',  n_jobs = -1, cv=5)
	grid_dt.fit(X_train, Y_train)
	DT_PARAM = grid_dt.cv_results_['params']
	print("DecisionTreeClassifier")
	print(grid_dt.best_score_)
	print(grid_dt.best_params_)
	print()
	
	# RF_GCV
	RF_CLF = RandomForestClassifier(n_jobs = -1)
	params = {'n_estimators':[130, 150, 170],
           	  'max_depth':[5, 10],
              'min_samples_leaf':[1, 2, 4],
              'min_samples_split':[2, 4, 8],
              'max_features':['auto', 'sqrt', 'log2'],

            }
	grid_cv = GridSearchCV(RF_CLF, param_grid = params, scoring='f1', n_jobs = -1, cv=5)
	grid_cv.fit(X_train,Y_train)
	RF_PARAM = grid_cv.best_params_
	print("RandomForestClassifier")
	print(grid_cv.best_score_)
	print(grid_cv.best_params_)
	print()
	return
	
	# XGB-GCV 
	XGB_CLF = xgboost.XGBClassifier()

	params = {'n_estimators':[100], 
					  'learning_rate':[0.05, 0.1, 0.15], 
					  'max_depth':[5],
					  'gamma':[0.2,0.25,0.3],
					  'subsample':[0.4, 0.5],
					  'colsample_bytree':[0.55,0.6,0.65],
					  'reg_lambda':[0.09,0.1,0.11],
					  'reg_alpha':[0, 0.01]}
	xgb_grid = GridSearchCV(XGB_CLF, param_grid=params, scoring='f1',  cv=5, refit=True)
	xgb_grid.fit(X_train, Y_train)
	XG_PARAM = xgb_grid.best_params_
	print("XGBClassifier")
	print(xgb_grid.best_score_)
	print(xgb_grid.best_params_)
	print()
	'''

	#########################################
	##### Hyper Parameter Tuning Result #####
	#########################################
	'''
	LogisticRegression
	0.9153606858988838
	{'max_iter': 175, 'penalty': 'none'}

	DecisionTreeClassifier
	0.8500169549759121
	{'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 0.01, 'min_samples_split': 6}

	RandomForestClassifier
	0.9902968584231788
	{'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}

	XGBClassifier
	0.9863600371906184
	{'colsample_bytree': 0.5, 'gamma': 0.15, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 0.1, 'subsample': 0.4}
	'''


	results = []
	for n in range(0,4):
		acc = 0
		pre = 0
		rec = 0
		f1 = 0
		roc = 0

		for _ in range(0, 50):
			a, b, c, d, e = Confuse_Matrix_Performance(X_train, X_test, Y_train, Y_test, n)
			acc += a
			pre += b
			rec += c
			f1 += d
			roc += e
		acc = round(acc/50, 4)
		pre = round(pre/50, 4)
		rec = round(rec/50, 4)
		f1 = round(f1/50, 4)
		roc = round(roc/50, 4)
		results.append([acc, pre, rec, f1, roc])

	for n in range(4):
		if(n==0):
			print("\n\n\n\nLogisticRegression")
		elif(n==1):
			print("\n\n\n\nDecisionTree")
		elif(n==2):
			print("\n\n\n\nRandomForest")
		else:
			print("\n\n\n\nXGBOOST")
		print(results[n])


def data_oversampling(df):
	smote = SMOTE()
	X = df.drop(['Pass/Fail'], axis=1)
	Y = df['Pass/Fail']
	X_train_over, Y_train_over = smote.fit_resample(X, list(Y))
	Y_train_over = pd.DataFrame(Y_train_over)
	return pd.concat([Y_train_over, X_train_over],axis=1)

def data_undersampling(df):
	save_col = df.loc[:, ['Time', 'Pass/Fail']]
	X = df.drop(['Time'], axis=1)
	Y = df['Pass/Fail']
	X_train_over, y_train_over = CondensedNearestNeighbour().fit_resample(X, list(Y))
	return pd.concat([save_col, X_train_over], axis=1)

## Fail데이터를 오버샘플링 한 결과물과
## Pass데이터를 언더샘플링을 한 결과물을 비교할 예정
'''
[Under_Random] RFE_MMS
0.8495 0.0367 0.0047 0.0082 0.4643
'''

'''
임시함수
'''
def visual(df):
	features = list(df.columns)
	plt.figure(figsize=(40,200))

	for i, feature in enumerate(features[2:]):
		print(i)
		print(feature)
		plt.subplot(12, 10, i+1)
		sns.boxplot(x="Pass/Fail",
					y=df[feature],
					data=df)
	#plt.show()
	#plt.savefig('[5]KBS_STD_PASS.png')

def visual2(df):
	features = list(df.columns)
	plt.figure(figsize=(40, 200))

	for i, feature in enumerate(features[2:]):
		print(feature)
		plt.subplot(12, 10, i+1)
		sns.boxplot(y=df[feature],
					data=df[feature])
	plt.savefig('[11]RFE_STD_FAIL.png')

# @param : 시작하고 싶은 전처리 단계
DataAnalytics(8)