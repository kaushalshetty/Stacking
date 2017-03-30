import numpy as np
from copy import copy
from sklearn.model_selection import KFold



class Stacking(object):
	def __init__(self,no_of_splits):
		super(Stacking, self).__init__()
		self.k = no_of_splits
		self.global_estimators=  []
		self.golbal_x = None
		self.global_y = None
		self.stackModel = None
		self.global_stack = None
		




	def fit(self,estimators,stacked_model,x,y):
		self.stackModel = stacked_model
		self.global_estimators = estimators
		self.global_x = x
		self.global_y = y
		kf = KFold(n_splits=  self.k)
		X_stack = None
		
		for train_index,test_index in kf.split(x,y):
			
				
			X_train,X_test = x[train_index],x[test_index]
			y_train,y_test = y[train_index],y[test_index]
			X_test_new_feats = None
			
			for i,estimator in enumerate(estimators):
				#print estimator
				estimator.fit(X_train,y_train)
				#print "x_test shape is",X_test.shape
				if i==0:
					X_test_new_feats = np.column_stack((X_test,np.asarray(estimator.predict(X_test),dtype=float)))
				else:
					 X_test_new_feats = np.column_stack((X_test_new_feats,np.asarray(estimator.predict(X_test),dtype=float)))
			if X_stack is None:
				#X_stack = np.empty([x_test_new_feats.shape[0],x_test_new_feats.shape[1]+len(estimators)],dtype=float)
				X_stack = X_test_new_feats 
			else:
				X_stack = np.vstack((X_stack,X_test_new_feats))
		
		model = copy(self.stackModel)
		print X_stack.shape,y.shape
		self.global_stack = X_stack
		print X_stack





	def predict(self,x_test):
		#x_test_meta = np.empty([x_test.shape[0],x_test.shape[1]+len(self.global_estimators)],dtype=float)
		for i,estimator in enumerate(self.global_estimators):
			estimator.fit(self.global_x,self.global_y)
			

			if i==0:
				x_test_meta = np.column_stack((x_test,np.asarray(estimator.predict(x_test),dtype=float)))
			else:
				x_test_meta = np.column_stack((x_test_meta,np.asarray(estimator.predict(x_test),dtype=float)))

		model = copy(self.stackModel)
		model.fit(self.global_stack,self.global_y)
		return np.asarray(model.predict(x_test_meta))



