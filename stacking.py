'''
Stacking classifier
Created on 3/4/2107
An ensemble-learning meta-classifier for stacking
Author: Kaushal Shetty
'''

import numpy as np
from copy import copy
from sklearn.model_selection import KFold



class Stacking(object):
	def __init__(self,estimators,stacked_model,use_prob=False,n_splits=5,verbose=0):
		"""
			Stacked Generalizer Classifier
			Trains a series of base models using K-fold cross-validation, then combines
			the predictions of each model into a set of features that are used to train
			a high-level classifier model. 
			Parameters
			-----------
			estmitators: list of classifier models
				Each model must have a .fit and .predict_proba/.predict method a'la
				sklearn
			stacked_model: object
				A classifier model used to aggregate the outputs of the trained base
				models. Must have a .fit and .predict_proba/.predict method
			use_prob: boolean 
				If True takes predict_proba as features else takes predict as features.
			n_splis: int
				The number of K-folds to use in =cross-validated model training
			verbose: 0 or 1
			
		"""
	
	
		super(Stacking, self).__init__()
		if len(estimators) == 0 or stacked_model is None:
			raise ValueError("Length of estimators must be greater than 0  and stack model connot be None type")

		if use_prob:
			for i in range(len(estimators)):
				if not hasattr(estimators[i],'predict_proba'):
					raise AttributeError(estimators[i]," has no attribute predict_proba. Either set use_prob to False or use a classifier that supports predict_proba")
		self.k = n_splits
		self.estimators=  estimators
		self.golbal_x = None
		self.global_y = None
		self.stackModel = stacked_model
		self.global_stack = None
		self.verbose = verbose
		self.use_prob = use_prob
		




	def fit(self,x,y):
	
		"""
    		
    		----------
    		estimators : list of base models, shape = len(estimators)
    		stacked_model : meta ensembling model (level 2 classifier)
    		x :  {array-like}, shape = [n_samples, n_features selected]
				 Training vectors, where n_samples is the number of samples and n_features is the number of features.
 
    		y  : array-like, shape = [n_samples]
    			 Target Values
    		Returns
    		-------
    		self : object 
    	"""
		if self.verbose>0:
			print("Base Estimators are ",self.estimators[0],self.estimators[1])
			print("Stacked model is",self.stackModel)
		
		
		self.global_x = x
		self.global_y = y
		kf = KFold(n_splits=  self.k)
		X_stack = None
		cnt = 0
		print self.use_prob
		for train_index,test_index in kf.split(x,y):
			cnt = cnt + 1
			
			if self.verbose>0:
				print(cnt ," iteration")
				print("Fitting...")	
			
			X_train,X_test = x[train_index],x[test_index]
			y_train,y_test = y[train_index],y[test_index]
			X_test_new_feats = None

			
			for i,estimator in enumerate(self.estimators):
				estimator.fit(X_train,y_train)
				
				if self.use_prob:
					if i==0:
						X_test_new_feats = np.column_stack((X_test,np.asarray(estimator.predict_proba(X_test),dtype=float)))
					else:
					 	X_test_new_feats = np.column_stack((X_test_new_feats,np.asarray(estimator.predict_proba(X_test),dtype=float)))

				else:
					if i==0:
						X_test_new_feats = np.column_stack((X_test,np.asarray(estimator.predict(X_test),dtype=float)))
					else:
					 	X_test_new_feats = np.column_stack((X_test_new_feats,np.asarray(estimator.predict(X_test),dtype=float)))
			
			if X_stack is None:

				X_stack = X_test_new_feats 
			else:
				X_stack = np.vstack((X_stack,X_test_new_feats))
		
		if self.verbose>0:
			print("Model Fitted.")

		model = copy(self.stackModel)
		print X_stack.shape
		self.global_stack = X_stack
		
		return self





	def predict(self,x_test):
	
		""" 
			Predict target values for X.
			Parameters
				x_test : {array-like}, shape = [n_samples, n_features]
				 vectors, where n_samples is the number of samples and n_features is the number of features.
			Returns
				labels : array-like, shape = [n_samples]
				Predicted class labels.

		"""
		if self.verbose>0:
			print("Predicting....")
		
		for i,estimator in enumerate(self.estimators):
			estimator.fit(self.global_x,self.global_y)
			if self.use_prob:
				if i==0:
					x_test_meta = np.column_stack((x_test,np.asarray(estimator.predict_proba(x_test),dtype=float)))
				else:
				 	x_test_meta = np.column_stack((x_test_meta,np.asarray(estimator.predict_proba(x_test),dtype=float)))

			else:
				if i==0:
					x_test_meta = np.column_stack((x_test,np.asarray(estimator.predict(x_test),dtype=float)))
				else:
					x_test_meta = np.column_stack((x_test_meta,np.asarray(estimator.predict(x_test),dtype=float)))

		
		model = copy(self.stackModel)
		model.fit(self.global_stack,self.global_y)
		
		if self.verbose>0:
			print("Done.")

		return np.asarray(model.predict(x_test_meta))



