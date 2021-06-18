import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# 추후 확장성을 고려하여 제작하였습니다. */
# 스케치 작업이니 작업 흐름을 봐주시면 좋겠습니다. #

def DataAnalytics(file_link):

	#원본 데이터 셋과 통계용 데이터 셋을 만듭니다. 
	raw_DF_original = pd.read_csv(file_link)
	raw_DF_statistic = raw_DF_original.describe().transpose()



	########## 데이터 구조 분석 과정 시작 ##########


	# 데이터 비율과 통계적 수치 확인
	## 시각화 시작 ##
	print("PASS / FAIL\n데이터 비율")
	# 이 부분은 file_link를 확인해야 하므로 추후 수정
	pie_ratio [0, 0]
	pie_labels ['PASS', 'FAIL']
	plt.pie(ratic=pie_ratio, labels=pie_labels, autopct='%.1f%%')
	plt.show()
	## 시각화 종료 ##

	# 데이터 프레임은 시각화가 필요 없으나, 그 내용을 가지고 무언가를 추론할 때 필요 */
	raw_DF_statistic.describe() 
	

	# 분석을 위해 숫자 데이터만 분리하고 Feature 번호를 만들어줌
	raw_DF_char = raw_DF_original.loc([:, ['Time', 'Pass/Fail']])
	raw_DF_inte = raw_DF_original.drop(['Time', 'Pass/Fail'], axis=1).add_prefix('F')
	raw_DF_original = pd.concat([raw_DF_char, raw_DF_inte], axis=1)

	raw_DF_original.head()


	########## 데이터 구조 분석 과정 종료 ##########

	#### 분석한 결과를 토대로 데이터를 가공 시작 ####

	########## 데이터 구조 가공 과정 시작 ##########


	# 통계상 무의미한 Feature 제거 #

	# 표준편차가 0인 Feature.
	# 여기서 전체 데이터와 표준편차가 0인 데이터를 시각화 하고
	# 가공 과정 단계마다 처리 과정 전 후의 차이를 시각화 데이터를 만들어 두면 좋을듯.
	remove_std = raw_DF_statistic[raw_DF_statistic['std'] == 0]

	# 표준편차 제거가 이루어진 데이터
	refine1_DF = raw_DF_original.drop(remove_std, axis=1)
	refine1_DF.describe()


	# 표준편차가 0에 가까운 Feature 확인
	refine1_DF_statistic = refine1_DF.describe().transpose()
	remove_std = refine1_DF_statistic[(refine1_DF_statistic['std'] > 0) & (refine1_DF_statistic['std'] < 0.005)].index

	# 과정 이해를 돕기 위해 변수를 남발하고 있습니다.
	# 최종 발표 시점에는 리팩토링을 진행하도록 하겠습니다.
	# 주석도 깔끔히 정리할 예정입니다... 
	refine2_DF = refine1_DF.drop(remove_std, axis=1)
	refine2_DF.describe()


	# 여기까지가 refined 2.2 까지의 과정입니다. 
