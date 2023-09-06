import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print(train_data.head())

# 결측치 찾기
print(train_data.isnull().sum())
# age 177개, Cabin 687개, embarked 2개
# cabin은 결측치가 너무 많아서 사용할 수 없을 듯
# age의 경우 중간값으로 처리, 평균값으로 처리, 제거 가능함
# embarked는 삭제하는게 나을 듯

# 항구와 pclass의 상관관계
correlation_table = pd.crosstab(train_data['Embarked'], train_data['Pclass'], margins=True)
print(correlation_table)
chi2, p, _, _ = chi2_contingency(correlation_table)
print("Chi-Square Statistic:", chi2)
print("p-value:", p)
# 범주형 변수 간의 상관관계는 카이제곱 독립성 검정, 크래머의 V를 사용함

# 자유도 설정 // (행의 개수 - 1) * (열의 개수 - 1)
degrees_of_freedom = (correlation_table.shape[0] - 1) * (correlation_table.shape[1] - 1)

# 유의수준 설정
significance_level = 0.05

# 임계값 설정
critical_value = chi2.ppf(1 - significance_level, degrees_of_freedom)

# 카이제곱 통계값과 임계값 비교
if chi2 > critical_value:
    print("크다")
else:
    print("작다")

# pclass와 생존 상관관계

# 성별과 생존 상관관계

# 나이와 생존 상관관계

# sibsp와 parch 생존 상관관계
