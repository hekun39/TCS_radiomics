# -*- coding: UTF-8 -*
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier

def my_df_scaler_collect(df,mean_list,std_list):
	for i in range(len(list(df.columns))):
		mean = mean_list[i]
		std = std_list[i]
		df[list(df.columns)[i]] = (df[list(df.columns)[i]] - mean+1e-7) / (std+1e-7)
	return df

if __name__ == "__main__":

	warnings.filterwarnings("ignore")
	result_table = []
	for hospital in ['TEST']:
		for stage in ['1','2','3','4','5']:
			temp = []
			temp.append(hospital)
			temp.append(stage)
			### load collect data
			collect_data = np.load('collect_data_'+stage+'_new.npy',allow_pickle = True).item()
			#### load test data
			X = np.load('L'+stage+'_'+hospital+'_final.npy')
			name_exp = X[0,:-1]
			X = X[1:].astype(np.float32)
			X = X[:,:-1]
			name_exp = np.arange(X.shape[1])
			name_exp = [str(it) for it in name_exp]
			df =  pd.DataFrame(X.astype(np.float32),columns = name_exp)
			##### for external test
			if True:
				loaded_ada_clf = joblib.load('model'+stage+'.pkl')
				df = my_df_scaler_collect(df,collect_data['mean_list'],collect_data['std_list'])
				select_data = df[collect_data['select_cols']]
				X_train = np.array(select_data)
				lst_ind = collect_data['lst_ind']
				X_train_t = X_train[:,lst_ind]
				X_train_t = np.squeeze(X_train_t,0)
				print('features_num',X_train_t.shape)
				y_score = loaded_ada_clf.predict_proba(X_train_t)[:,1]
				print('predict',y_score)
				for score in y_score:
					temp.append(score)
			result_table.append(temp)
	print(result_table)
