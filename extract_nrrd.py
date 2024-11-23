import radiomics
import radiomics.featureextractor as FEE
import numpy
import SimpleITK as sitk
import os
import numpy as np
def extract_one2arr_label(pathx,pathy,extractor,arr,loc,target, label,if_first =False):
	result = extractor.execute(pathx,pathy,label = label)
	cols = 0
	num = 0
	newarr = []
	if if_first:
		for key,value in result.items():
			if num>36:
				newarr.append(key)
			num +=1
		newarr.append('target')
	else:
		for key,value in result.items():
			if num>36:
				newarr.append(value)
			num += 1
		newarr.append(label)
	newarr = np.array(newarr) 
	arr.append(newarr)

	return arr
params = 'params2.yaml'
extractor = FEE.RadiomicsFeatureExtractor(params)
loc = 0
##################################
first_path = './test'  ###change hospital path
#################################
target = 1
patient_name = os.listdir(first_path)
if_first = True
for num_patint in range(len(patient_name)):
	S_patient_name = patient_name[num_patint]
	second_path = first_path + '/' + S_patient_name
	path_T2 = second_path+ '/t2.nrrd'
	path_seg = second_path + '/seg.nrrd'
	imgy = sitk.ReadImage(path_seg)
	arry = sitk.GetArrayFromImage(imgy)
	for i in range(1,6):
		lst = np.where(arry == i)
		if len(lst[0])>0:
			try:
				if if_first:
					arrT2 = extract_one2arr_label(path_T2,path_seg,extractor,arrT2,loc,target,label = i,if_first=if_first)
					loc += 1
					if_first = False
				arrT2 = extract_one2arr_label(path_T2,path_seg,extractor,arrT2,loc,target,label = i,if_first=False)

			except Exception as re:
				print(re)
				...
			loc += 1
			print(S_patient_name,'label ', i, path_T2, path_seg)
arrT2 = np.array(arrT2)
np.save('nrrds_TEST_final.npy',arrT2)
path = 'nrrds_TEST_final.npy'
a = np.load(path)
name = a[0,:]
a = a[1:]
L1 = a[0::5,:]
L2 = a[1::5,:]
L3 = a[2::5,:]
L4 = a[3::5,:]
L5 = a[4::5,:]
name = name.reshape(1,-1)
L1N = np.concatenate([name,L1],axis = 0)
L2N = np.concatenate([name,L2],axis = 0)
L3N = np.concatenate([name,L3],axis = 0)
L4N = np.concatenate([name,L4],axis = 0)
L5N = np.concatenate([name,L5],axis = 0)
np.save('L1_TEST_final.npy',L1N)
np.save('L2_TEST_final.npy',L2N)
np.save('L3_TEST_final.npy',L3N)
np.save('L4_TEST_final.npy',L4N)
np.save('L5_TEST_final.npy',L5N)



