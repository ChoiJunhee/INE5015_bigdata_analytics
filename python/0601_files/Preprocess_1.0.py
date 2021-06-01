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

def DataAnalytics(steps):
	pass


'''
@param - file_link : Raw CSV 파일의 주소
@return - CSV 파일의 Dataframe
'''
def get_data(file_link):
	pass

'''
@param - df : 가공할 데이터 프레임
@param - num : 기준이 될 표준편차
@return - 기준 이하의 표준편차를 가진 Feature가 제거된 데이터 프레임
'''
def data_std_remove(df, num):
	pass

'''
@param - df : 다중공선성을 제거할 데이터 프레임
@param - num : 기준이 될 상관관계 지수
@return - 상관관계가 높은 Feature가 제거된 데이터 프레임
'''
def correlation_remove(df, per):
	pass

'''
@param - df : 결측치를 제거할 데이터 프레임
@param - per : 기준이 될 결측치 비율
@return - 기준 이상의 결측치를 갖는 Feature가 삭제된 데이터 프레임
'''
def missing_value_processing(df, per):
	pass

def outlier_processing(df, per):
	pass

def set_data_scale(df, num):
	pass

def feature_selection(df, num):
	pass

def data_oversampling(df, num):
	pass







############################ To be Updated ##########################
## 1. 미정                                                          ##
## 2. 미정                                                          ##
#####################################################################