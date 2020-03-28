import sinc_PrepareDataset as PD
import theano
import lasagne
import theano.tensor as T
from theano import shared
# import sinc_Network as Network
from time import gmtime, strftime
import os
import numpy as np
import parser
import sys

log_dir = 'log/'
# division = 150
# fold = 1
# augment = True

my_map = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store',
'home', 'beach', 'library', 'metro_station', 'office', 'residential_area', 
'train', 'tram', 'park']

def test(network_name,depth,dataset_name,nonL,fold,division,augment,patience,representation,epochs,learning_rate,batch_size,dropout,growth_rate):
	# batch_size = 128
	num_classes = 15
	
	#tensor variables for inputs and targets
	input_var = T.tensor3('inputs')
	target_var = T.fmatrix('targets')

	# if augmentation=='True':
	# 	augment = True
	# else :
	# 	augment = False

	X_test,y_test, num_replicates = PD.load_testset(dataset_name,fold,division,representation)
	X_val_fold_1,y_val_fold1, num_replicates = PD.load_valset(dataset_name,1,division,representation)
	X_val_fold_2,y_val_fold2, num_replicates = PD.load_valset(dataset_name,2,division,representation)
	X_val_fold_3,y_val_fold3, num_replicates = PD.load_valset(dataset_name,3,division,representation)
	X_val_fold_4,y_val_fold4, num_replicates = PD.load_valset(dataset_name,4,division,representation)




	data_len = X_test.shape[1]/division

	if network_name=='Sinc':
		import sinc_Network as Network
	elif network_name=='SampleCNN':
		import SCNN_Network as Network
	elif network_name=='SampleCNN+Sinc':
		import SCNN_Sinc_Network as Network
	elif network_name=='DSNP+Sinc':
		import DSNP_Network as Network


	network = Network.build_cnn(data_len,input_var=input_var)
	print data_len

	#loss for validation/testing, note that deterministic=true means we disable droput layers for test/eval
	test_prediction = lasagne.layers.get_output(network, deterministic = True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	#computing the accuracy
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),T.argmax(target_var,axis=1)), dtype=theano.config.floatX)

	#test and eval (no updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])
	


	#training is over, compute the test loss
	
	test_pred = np.zeros([X_test.shape[0],num_classes])
	test_correct_pred = np.zeros(X_test.shape[0])
	
	# visit = np.zeros(X_test.shape[0])


	directory = 'testModels/'
	for model in os.listdir(directory):
		if network_name not in model:
			continue
		print model
		with np.load(directory + model) as f:
			# print f['arr_0'].shape
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			# print len(param_values)

		lasagne.layers.set_all_param_values(network, param_values)
		test_err = 0
		test_acc = 0
		test_batches = 0
		test_pred_model = np.zeros([X_test.shape[0],num_classes])

		

		if 'Left' in model:
			rep = 'Left'

		if 'Right' in model:
			rep = 'Right'

		if 'Mid' in model:
			rep = 'Mid'	

		if 'Side' in model:
			rep = 'Side'

		if 'fold1' in model:
			X_val = X_val_fold_1
			y_val = y_val_fold1

		if 'fold2' in model:
			X_val = X_val_fold_2
			y_val = y_val_fold2

		if 'fold3' in model:
			X_val = X_val_fold_3
			y_val = y_val_fold3

		if 'fold4' in model:
			X_val = X_val_fold_4
			y_val = y_val_fold4

		val_err = 0
		val_acc = 0
		val_batches = 0
		val_pred_model = np.zeros([X_val.shape[0],num_classes])
		val_pred = np.zeros([X_val.shape[0],num_classes])
		val_correct_pred = np.zeros(X_val.shape[0])


		print("Loading data...")
		assert (rep==representation)


		batches = PD.iterate_minibatches(X_val, y_val, batch_size,division=division,num_samples=division,
		shuffle=False,noise = False,mixup=False,val=True)
		num_data=division
		for batch in batches:
			# print " test_batches:\t{}".format(test_batches)
			inputs, targets, indices = batch
			val_error, val_accuracy, prediction= val_fn(inputs, targets)
			val_err += val_error
			val_acc += val_accuracy
			val_batches += 1
			for i in range(len(indices)):
				# visit[indices[i]/num_data] += 1
				# val_pred[indices[i]/num_data] += prediction[i]
				val_pred_model[indices[i]/num_data] += prediction[i]
				val_correct_pred[indices[i]/num_data] = np.argmax(targets[i])

		real_val_accuracy_model = np.mean(np.equal(np.argmax(val_pred_model, axis=1),val_correct_pred))
		print real_val_accuracy_model
		print "final results on model " + model[:-4] + ":"
		print " val loss:\t{:.6f}".format(val_err/val_batches)
		print " val accuracy:\t{:.2f} %".format(val_acc/val_batches *100)
		print " resl val accuracy on this model:\t{:.2f} %".format(real_val_accuracy_model*100)
		# X_test, y_test,num_data = PD.load_testset(dataset_name,fold,division,representation,False)

		batches = PD.iterate_minibatches(X_test, y_test, batch_size,division=division,num_samples=division,
		shuffle=False,noise = False,mixup=False,val=True)
		num_data=division
		for batch in batches:
			# print " test_batches:\t{}".format(test_batches)
			inputs, targets, indices = batch
			test_error, test_accuracy, prediction= val_fn(inputs, targets)
			test_err += test_error
			test_acc += test_accuracy
			test_batches += 1
			for i in range(len(indices)):
				# visit[indices[i]/num_data] += 1
				test_pred[indices[i]/num_data] += prediction[i]
				test_pred_model[indices[i]/num_data] += prediction[i]
				test_correct_pred[indices[i]/num_data] = np.argmax(targets[i])

		real_test_accuracy_model = np.mean(np.equal(np.argmax(test_pred_model, axis=1),test_correct_pred))
		print real_test_accuracy_model
		print "final results on model " + model[:-4] + ":"
		print " test loss:\t{:.6f}".format(test_err/test_batches)
		print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
		print " resl test accuracy on this model:\t{:.2f} %".format(real_test_accuracy_model*100)
		print "----------------------------------------------"


		

	# for v in visit:
	# 	if v != fold*division:
	# 		print v
	test_predicted_label = np.argmax(test_pred, axis=1)
	real_test_accuracy = np.mean(np.equal(np.argmax(test_pred, axis=1),test_correct_pred))

	print "final results:"
	print " real test accuracy:\t{:.2f} %".format(real_test_accuracy*100)

def main():
    # parse command line
    my_parser = parser.opts_parser()
    args = my_parser.parse_args()

    # run
    test(**vars(args))


if __name__ == "__main__":
	main()

# dataset_name = sys.argv[1]
# fold = int(sys.argv[2])
# division = int(sys.argv[3])
# # patience = sys.argv[4]
# rep = sys.argv[4]
# augmentation = sys.argv[4]
# test()s