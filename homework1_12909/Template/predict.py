import os
import os.path
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()

# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

if args.model_name == 'GaussianNB':
	pass 
elif args.model_name == 'LogisticRegression':
	pass
elif args.model_name == 'DecisionTreeClassifier':
	# load the model

	# model = DecisionTreeClassifier(  ...  )

	# save the predictions in a text file with the predicted clasdIDs , one in a new line 
else:
	raise Exception("Invald Model name")
