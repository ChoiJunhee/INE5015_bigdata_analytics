import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path


def close_std_zero_remove(raw_df, num):
	# 이 함수는 표준편차가 num미만인 데이터를 제거합니다.
	raw_df_trans = raw_df.describe().transpose()

	remove_std = raw_df_trans[raw_df_trans['std'] <= num].index
	result = raw_df.drop(remove_std, axis=1)

	return result


file = pd.read_csv('./rf3/rf3_c30_m60_pass.csv')
test = close_std_zero_remove(file, 5.0);
sns.stripplot(data=test, jitter=True, size=1);
plt.show()