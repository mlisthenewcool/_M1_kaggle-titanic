#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np
import math

def get_data():
	train_df = pd.read_csv('./data/train.csv')
	test_df = pd.read_csv('./data/test.csv')
    
	labels = train_df['Survived']
	#train_df.drop(['Survived'], 1, inplace=True)
	return train_df, test_df, labels

def kaggle_submission(x_test, y_test, submission_name):
	df = pd.DataFrame()
	df['PassengerId'] = x_test['PassengerId']
	df['Survived'] = y_test
	df.to_csv(submission_name, index=False)

###############################################################################
def compute_family_size(df):
	df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
	df['IsAlone'] = df['FamilySize'].map(lambda x: 1 if x == 1 else 0)
	df['SmallFamily'] = df['FamilySize'].map(lambda x: 1 if 2 <= x <= 4 else 0)
	df['BigFamily'] = df['FamilySize'].map(lambda x: 1 if x >= 5 else 0)
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
			if grouped_median.iloc[pos]['Sex'] == row['Sex'] and \
               grouped_median.iloc[pos]['Title'] == row['Title'] and \
               grouped_median.iloc[pos]['Pclass'] == row['Pclass']:
				return grouped_median.iloc[pos]['Age']

	df['Age'] = df.apply(fill_age, axis=1)

	return df

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

def compute_lastnames(df):
    # getting last names
    df['LastName'] = df['Name'].apply(lambda x: str.split(x, ',')[0])
    return df

def compute_fares(df):
	# missing values will be replaced with median
	df['Fare'].fillna(df['Fare'].median(), inplace=True)
	return df

def compute_family_groups(df):
    # les femmes sont regroupées
    #df[df['Title'] == 'Mrs' or 'Miss']['Title'] = 'Woman'
    
    # les enfants sont regroupés
    #df[df['Title'] == 'Master']['Title'] = 'Boy'
    
    ######
    
    def find_group(row):
        if row['Title'] in ['Mr', 'Royalty', 'Officer']:
            return 'noGroup'
        elif row['IsAlone'] == 1:
            return 'noGroup'
        return 'grouped'
    
    # les hommes n'ont pas de groupe spécifique
    # les officiers et nobles n'ont plus
    #df[df['Title'] == 'Mr']['Group'] = 'none'
    
    # taille des familles
    #families = df.groupby('LastName').size()
    
    # les personnes seules n'ont pas de groupe
    #df[df['IsAlone'] == 1]['Group'] = 'none'
    
    """
    groups = {
        "Mr" : "noGroup",
        "Royalty" : "noGroup",
        "Officer" : "noGroup"
    }    
    df['Group'] = df['Title']
    df['Group'] = df.Group.map(groups)
    """
    
    
    # groupons les familles par survie
    df['Group'] = 'grouped'
    df['Group'] = df.apply(find_group, axis=1)
    
    #families = df.groupby(['LastName']).mean()
    #print(df.groupby(['LastName', 'Survived']).mean())
    
    groups = df[df['Group'] != 'noGroup'].groupby('LastName')
    
    #print(groups[groups['Survived'] == 0])
    for group in groups:
        if group[1] == 0: 
            print(group[0])
    
    return df
    
def compute_groups(df):
    families = df.groupby(['LastName']).mean()
    
    print(families.columns)
    
    #print(df[df['Group'] != 'none'])
    """
    survived_frequence = df.groupby('LastName').transform(lambda x: pd.Series(x.Survability.length))
    
    print(survived_frequence)
    
    def find_group(row):
        if row['Title'] == 'Officer' or 'Royalty' or 'Mr':
            return 'noGroup'
        else:
            return 'toDo'
    
    df['Group'] = df.apply(find_group, axis=1)
    """
###############################################################################

def gender_model(row):
	return 1 if row['Sex'] == 'female' else 0


def find_groups_against_the_gender_odds(df):
    """ Les femmes et les enfants d'abord !
    """
    hommes = df[df['Sex'] == 'male']
    femmes = df[df['Sex'] == 'female']
    
    # on cherche à trouver un âge à partir duquel les hommes survivent plus
    # qu'ils ne meurent
    a = sns.FacetGrid(hommes, hue='Survived', aspect=4)
    a.map(sns.kdeplot, 'Age', shade=True)
    a.set(xlim=(hommes['Age'].min(), hommes['Age'].max()))
    a.add_legend()
    
    tous = [0, 0, 0, 0, 0]
    morts = [0, 0, 0, 0, 0]
    ages = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26]
    
    ind_age = 0
    
    for age in ages:
        for ind_homme in range(len(hommes)):
            if hommes.iloc[ind_homme]['Age'] <= age:
                tous[ind_age] += 1
                if hommes.iloc[ind_homme]['Survived'] != 1:
                    morts[ind_age] += 1
        print(age, morts[ind_age] / tous[ind_age])
    
    # on cherche à trouver une catégorie sociale à partir de laquelle les
    # femmes meurent plus qu'elles ne survivent


# appliquer uniquement le modèle de genre
# test['Survived'] = test.apply(gender_model, axis=1)
train, test, labels = get_data()

train = compute_family_size(train)
train = compute_names_to_titles(train)
train = compute_missing_ages(train)
train = compute_lastnames(train)
#train = compute_groups(train)
train = compute_family_groups(train)
#train = compute_groups(train)

#find_groups_against_the_gender_odds(train)