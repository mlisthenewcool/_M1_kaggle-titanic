#!/usr/bin/env pyth
# -*- coding: utf-8 -*-

"""
Package description
"""

__author__    = "Hippolyte Debernardi"
__copyright__ = "Copyright 2018, Hippolyte Debernardi"
__license__   = "GPL"
__version__   = "0.0.1"
__email__     = "contact@hippolyte-debernardi.com"


def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

import math
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

def get_data():
	train = pd.read_csv('./train.csv')
	test = pd.read_csv('./test.csv')
	
	#y = train['Survived']
	train.drop(['Survived'], 1, inplace=True)
	
	df = train.append(test)
	df.reset_index(inplace=True)
	df.drop(['index', 'PassengerId'], inplace=True, axis=1)
	
	return df

def feature_status(feature):
	print('Processing {} done.'.format(feature))

def compute_names_to_titles(df):
	titles = {
		"Capt": "Officer",
		"Col": "Officer",
		"Major": "Officer",
		"Dr": "Officer",
		"Rev": "Officer",
		"Jonkheer": "Royalty",
		"Don": "Royalty",
		"Dona": "Royalty",
		"Sir": "Royalty",
		"the Countess": "Royalty",
		"Lady": "Royalty",
		"Mme": "Mrs",
		"Ms": "Mrs",
		"Mrs": "Mrs",
		"Mlle": "Miss",
		"Miss": "Miss",
		"Mr": "Mr",
		"Master": "Master"
	}

	df['Title'] = df['Name'].map(
		lambda name: name.split(',')[1].split('.')[0].strip())
	df['Title'] = df.Title.map(titles)

	return df

def compute_missing_ages(df):
	# missing values will be replaced with a combination of 3 features
	# we take the median according to sex, pclass and title columns
	grouped = df.groupby(['Sex', 'Pclass', 'Title'])
	grouped_median = grouped.median()
	grouped_median = grouped_median.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

	def fill_age(row):
		if not math.isnan(row['Age']):
			return row['Age']

		for pos in range(len(grouped_median)):
			if (grouped_median.iloc[pos]['Sex'] == row['Sex']) and (grouped_median.iloc[pos]['Title'] == row['Title']) and (grouped_median.iloc[pos]['Pclass'] == row['Pclass']):
				return grouped_median.iloc[pos]['Age']

	df['Age'] = df.apply(fill_age, axis=1)

	return df

def compute_fares(df):
	# missing values will be replaced with median
	df['Fare'].fillna(df['Fare'].median(), inplace=True)
	return df

def compute_embarked(df):
	# missing values will be replaced with most frequent value
	df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
	return df

def compute_cabin(df):
	# missing values will be replaced with U as unknown
	df['Cabin'].fillna('U', inplace=True)
	df['Cabin'] = df['Cabin'].map(lambda c: c[0])

	return df

def compute_sex(df):
	df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
	return df

def compute_ticket(df):

	def extract_data_from_ticket(ticket):
		ticket = ticket.replace('.', '')
		ticket = ticket.replace('/', '')
		ticket = ticket.split()
		ticket = map(lambda t: t.strip(), ticket)
		ticket = list(filter(lambda t: not t.isdigit(), ticket))

		if len(ticket) > 0:
			return ticket[0]
		return 'XXX'

	df['Ticket'] = df['Ticket'].map(extract_data_from_ticket)
	return df

def compute_family(df):
	df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
	df['IsAlone'] = df['FamilySize'].map(lambda x: 1 if x == 1 else 0)
	df['SmallFamily'] = df['FamilySize'].map(lambda x: 1 if 2 <= x <= 4 else 0)
	df['BigFamily'] = df['FamilySize'].map(lambda x: 1 if x >= 5 else 0)
	return df

class FeatureExtractor():
	def __init__(self):
		pass

	def transform(self, X_df):
		# 0. COPIE du jeu de donnees
		df = X_df.copy(deep=True)
		
		#df['Age'] = df['Age'].fillna(df['Age'].median())

		# 1. CREATION
		compute_names_to_titles(df)
		feature_status('Title')

		compute_missing_ages(df)
		feature_status('Age')

		compute_fares(df)
		feature_status('Fare')

		compute_embarked(df)
		feature_status('Embarked')

		compute_cabin(df)
		feature_status('Cabin')
		
		compute_sex(df)
		feature_status('Sex')
		
		compute_family(df)
		feature_status('Family')
		
		compute_ticket(df)
		feature_status('Ticket')

		# 2. TRANSFORMATION pour les algos de ML
		cabins = df['Cabin'].astype('category', categories=list('ABCDEFGU'))
		
		df_new = pd.concat([
			df.get(['Age', 'Fare', 'Sex', 'FamilySize', 'IsAlone', 'SmallFamily', 'BigFamily']),
			pd.get_dummies(df.Title, prefix='Title'),
			pd.get_dummies(df.Embarked, prefix='Embarked'),
			pd.get_dummies(cabins, prefix='Cabin'),
			pd.get_dummies(df.Pclass, prefix='Pclass'),
			pd.get_dummies(df.Ticket, prefix='Ticket')
		], axis=1)

		#print(df_new.columns)
		print('-'*80)

		return df_new

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

class Classifier():
	def __init__(self):
		clf1 = RandomForestClassifier(
			max_depth=7,
			n_estimators=250)

		clf2 = GradientBoostingClassifier(
			n_estimators=200,
			learning_rate=0.05,
			subsample=0.6)

		clf3 = BaggingClassifier(
			RandomForestClassifier(
				criterion='gini', max_depth=20, n_estimators=100, max_features='log2'),
			max_samples=0.75, max_features=0.5, n_estimators=20)

		xgb_paramaters = {'subsample' : [0.7], 'min_child_weight' : [1], 'max_depth' : [3], 'learning_rate' : [0.1], 'n_estimators' : [100], 'n_jobs' : [-1], 'random_state' : [1]}
		clf4 = GridSearchCV(XGBClassifier(), xgb_paramaters, n_jobs=-1, cv=8)

		eclf = VotingClassifier(
			estimators=[
				('rf', clf1),
				('gbc', clf2)
				#('bc', clf3)],
				#('xgb', clf4)],
			],
			voting='soft')

		self.clf = Pipeline([
		  ('imputer', Imputer(strategy='median')),
		  ('classifier', eclf)
		])

		#features = pd.DataFrame()
		#features['feature'] = train.columns
		#features['importance'] = self.clf._get_params_
		#features.sort_values(by=['importance'], ascending=True, inplace=True)
		#print(self.clf.get_params())
	
	def compute_score(self, X, y, cv=8, scoring='accuracy'):
		return np.mean(cross_val_score(self.clf, X, y, cv=cv, scoring=scoring))

	def fit(self, X, y):
		self.clf.fit(X, y)

	def predict_proba(self, X):
		return self.clf.predict_proba(X)
	
	def predict(self, X):
		return self.clf.predict(X)

def kaggle_submission():
	y_labels = pd.read_csv('./train.csv', usecols=['Survived'])['Survived'].values
	
	fe = FeatureExtractor()
	data = fe.transform(get_data())
	
	print(data.shape)
	
	train = data.iloc[:891]
	test = data.iloc[891:]
	
	clf = Classifier()
	clf.fit(train, y_labels)
	
	predicts = clf.predict(test).astype(int)
	
	df_final = pd.DataFrame()
	df_final['PassengerId'] = pd.read_csv('./test.csv')['PassengerId']
	df_final['Survived'] = predicts
	df_final[['PassengerId', 'Survived']].to_csv('first_submission.csv', index=False)
	
	#features = pd.DataFrame()
	#features['features'] = train.columns
	#features['importance'] = clf.clf.feature_importances_
	
	print(clf.compute_score(train, y_labels))

if __name__ == '__main__':
	kaggle_submission()
