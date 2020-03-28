import sinc_PrepareDataset as PD
import theano
import lasagne
import theano.tensor as T
from theano import shared
# import sinc_Network as Network
from time import gmtime, strftime
import os
import numpy as np
import sys
# import progress
import background as bg
import parser 
from helpers import print_net_architecture

log_dir = 'log/'
# division = 1
# fold = 1
# patience='False'
# representation = 'Mid'
# augment = True

my_map = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store',
'home', 'beach', 'library', 'metro_station', 'office', 'residential_area', 
'train', 'tram', 'park']

num_classes = 15




def train(network_name,depth,dataset_name,nonL,fold,division,augment,patience,representation,epochs,learning_rate,batch_size,dropout,growth_rate):
	
	def iterate(name,data,label,num_replicates,batch_size,shuffle=False):
		num_samples = division
		num_replicates = num_samples
		train_err = 0
		train_acc = 0.0
		train_batches = 0
		visit = np.zeros(data.shape[0])
		train_pred = np.zeros([data.shape[0],num_classes])
		# print 'train_pred = ' + str(train_pred.shape)
		train_voting_pred = np.zeros([data.shape[0],num_classes])
		train_correct_pred = np.zeros(data.shape[0])

		noise=False
		mixup = False
		val=True
		if name=='train':
			noise=True
			mixup=False
			val=False

		batches = PD.iterate_minibatches(data, label, batch_size,division=division,num_samples=num_samples,
		shuffle=shuffle,noise = noise,mixup=mixup,val=val)
		batches=bg.generate_in_background(batches)
		# print sum(1 for x in batches)
		# if augment and name == 'train':
		# 	batches = PD.augment_minibatches(batches)
		# 	batches = bg.generate_in_background(batches)
			# print sum(1 for x in batches)

		for batch_cnt,batch in enumerate(batches):
			best_model = lasagne.layers.get_all_param_values(network)
			# print best_model.shape
			# if batch_cnt<10:
			# 	print batch_cnt
			# 	print len(best_model)
			# 	print best_model[0]
			# print " train_batches:\t{}".format(train_batches)
			inputs, targets, indices = batch
			
			if name == 'train':
				train_error, train_accuracy, prediction = train_fn(inputs, targets)
			else :
				train_error, train_accuracy, prediction = val_fn(inputs, targets)
			train_err += train_error
			train_acc += train_accuracy
			train_batches += 1

			# if name != 'train':
			for i in range(len(targets)):
				train_pred[indices[i]/num_replicates] += prediction[i]
				train_voting_pred[indices[i]/num_replicates][np.argmax(prediction[i])] += 1
				train_correct_pred[indices[i]/num_replicates] = np.argmax(targets[i])
				visit[indices[i]/num_replicates] += 1



		print name + " loss:\t{:.6f}".format(train_err/ train_batches)
		# print " l2 loss:\t{:.6f}".format(l2_loss)
		print name + " accuracy:\t{:.2f} %".format(train_acc/train_batches *100)
		# if name=='train':
		# 	print '\n'
		# l2_loss = l2_fn()
		
		train_predicted_label = np.argmax(train_pred, axis=1)
		real_train_accuracy = np.mean(np.equal(np.argmax(train_pred, axis=1),train_correct_pred))
		voting_train_accuracy = np.mean(np.equal(np.argmax(train_voting_pred, axis=1),train_correct_pred))
		print 	"real " + name + " accuracy:\t{:.2f} %\n".format(real_train_accuracy *100)
		
		

		return float(train_err/ train_batches),float(real_train_accuracy*100),float(train_acc/train_batches *100)
		

	date = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
	model_name = 'final_models/' + date + '_' + network_name + '_' + dataset_name +'-fold' + str(fold)+ '_wave_model_' + str(fold) \
	 + '_' + str(division) + '_' + str(patience) + '_' + representation + '_' + str(augment) + '.npz'
	num_classes = 15
	print date
	
	print "Instantiating network..."
	#tensor variables for inputs and targets
	input_var = T.tensor3('inputs')
	target_var = T.fmatrix('targets')
	# min_learning_rate = 0.000016
	best_acc = 0
	max_patience = 20
	# decrease_factor = 0.2
	current_patience = 0
	my_train_buffer = np.empty(max_patience+2, dtype=str)
	my_validation_buffer = np.empty(max_patience+2, dtype=str)

	print "Loading data..."
	X_train, y_train, X_val, y_val, X_test,y_test, num_replicates = PD.load_dataset(dataset_name,fold,division,representation)
	data_len=X_train.shape[1]/division

	if network_name=='Sinc':
		import sinc_Network as Network
	elif network_name=='SampleCNN':
		import SCNN_Network as Network
	elif network_name=='SampleCNN+Sinc':
		import SCNN_Sinc_Network as Network
	elif network_name=='DSNP+Sinc':
		import DSNP_Network as Network




	network = Network.build_cnn(data_len,input_var=input_var)
	print_net_architecture(network,detailed=True)
	print "%d layers with weights, %d parameters" % (sum(hasattr(l, 'W') for l in lasagne.layers.get_all_layers(network)),
		lasagne.layers.count_params(network, trainable=True))

	
	# num_train_replicates = num_replicates
	# num_val_replicates = num_replicates
	# num_test_replicates = num_replicates
	assert (num_replicates==division)
	assert (data_len==X_test.shape[1]/division)

	print X_train.shape,X_val.shape,X_test.shape

	

	print "Compiling training function..."
	#the final prediction of the network
	prediction = lasagne.layers.get_output(network)

	#define the loss
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
	# l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
	# network, lasagne.regularization.l2, {'trainable': True})

	learning_rate = theano.shared(lasagne.utils.floatX(learning_rate), name='learning_rate')
	
	#computing the accuracy
	train_acc = T.mean(T.eq(T.argmax(prediction, axis=1),T.argmax(target_var,axis=1)), dtype=theano.config.floatX)

	#update the parameters based on SGD with nesterov momentum
	params = lasagne.layers.get_all_params(network, trainable = True)
	if network_name=='Sinc' or 'DSNP+Sinc':
		updates = lasagne.updates.rmsprop(
			loss, params, learning_rate=learning_rate, rho=0.95,epsilon=1e-7)
	if network_name=='SampleCNN' or network_name=='SampleCNN+Sinc':
		updates = lasagne.updates.nesterov_momentum(
		loss, params, learning_rate=learning_rate, momentum=0.9)

	#perform the training
	train_fn = theano.function([input_var, target_var], [loss, train_acc, prediction], updates = updates)
	# l2_fn = theano.function([], l2_loss)

	print "Compiling testing function..."
	#loss for validation/testing, note that deterministic=true means we disable droput layers for test/eval
	test_prediction = lasagne.layers.get_output(network, deterministic = True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	#computing the accuracy
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),T.argmax(target_var,axis=1)), dtype=theano.config.floatX)
	#test and eval (no updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])


	
	train_log = log_dir + 'final_' + date+ 'fold_' + str(fold) + '_' +network_name +'_training_log.txt'
	validation_log = log_dir + 'final_' + date +  'fold_' + str(fold) + '_' + network_name + '_validation_log.txt'
	test_log = log_dir + date + 'final_' + 'fold_' + str(fold) + '_' + network_name +'_test_log.txt'
	best_train_log = log_dir + 'final_' + date+ 'best_fold_' + str(fold) +'_training_log.txt'
	best_validation_log = log_dir + 'final_' + date +  'best_fold_' + str(fold) +'_validation_log.txt'

	#iterate over epoches
	print 'training' + network_name + ' on the dataset ' + dataset_name +  ' divided into ' + str(division) + " segments and fold number: " + str(fold) + " learning_rate = "+ str(learning_rate.get_value()) + " patience = " + str(patience) + 'augmentation = ' + str(augment) + ' representation = ' + representation
	
	# best_model = lasagne.layers.get_all_param_values(network)
	
	for epoch in range(epochs):
		print " Epoch {} of {} \n".format(epoch + 1, epochs)
		train_err,real_train_acc, train_acc = iterate('train', X_train,y_train,num_replicates,batch_size,shuffle=True)
		val_err, real_val_acc, val_acc = iterate('validation', X_val,y_val,num_replicates,batch_size,shuffle=False)
		test_err, real_test_acc, test_acc = iterate('test', X_test,y_test,num_replicates,batch_size,shuffle=False)
		with open(train_log, 'a') as text_file:
			text_file.write("Epoch {} of {} fold: {} training_loss: {:.6f}  ,real_train_accuracy: {:.2f} ,train_accuracy: {:.2f} %\n\n".format(epoch, epochs,fold,train_err, real_train_acc, train_acc))

		with open(validation_log, 'a') as text_file:
			text_file.write("Epoch {} of {} fold: {} validation_loss: {:.6f}  ,real_validation_accuracy: {:.2f} ,validation_accuracy: {:.2f} %\n\n".format(epoch, epochs,fold,val_err, real_val_acc,val_acc))


		if(patience==True):
			#updating the best model
			epoch_acc = val_acc
			if(epoch_acc > best_acc):
				print 'best model updated...'
				
				#update the best model
				best_acc = epoch_acc
				# best_model = lasagne.layers.get_all_param_values(network)
				np.savez(model_name, *lasagne.layers.get_all_param_values(network))
				for i in range(current_patience+1):
					with open(best_train_log, 'a') as text_file:
						text_file.write(my_train_buffer[i])
					with open(best_validation_log, 'a') as text_file:
						text_file.write(my_validation_buffer[i])

				my_train_buffer = np.empty(max_patience+2, dtype=str)
				my_validation_buffer = np.empty(max_patience+2, dtype=str)
				current_patience = 0
			else:
				print 'current_patience increased to: ' + str(current_patience)
				print 'learning_rate = ' + str(learning_rate.get_value())
				current_patience += 1
				if(current_patience>max_patience):
					print 'resumed from best model with accuracy : ' + str(best_acc)
					my_train_buffer = np.empty(max_patience+2, dtype=str)
					my_validation_buffer = np.empty(max_patience+2, dtype=str)
					current_patience = 0
					if learning_rate.get_value()>lasagne.utils.floatX(0.000001):
						learning_rate.set_value(learning_rate.get_value()* lasagne.utils.floatX(0.5))

					
					with np.load(model_name) as f:
						# print f['arr_0'].shape
						param_values = [f['arr_%d' % i] for i in range(len(f.files))]
						# print len(param_values)

					lasagne.layers.set_all_param_values(network, param_values)


		print "---------------------------------------------------------------------------------------"
		# with open(train_log, 'a') as text_file:
		# 	text_file.write("Epoch {} of {} fold: {} training_loss: {:.6f} training_accuracy: {:.2f} % ,real_train_accuracy: {:.2f} %\n\n".format(epoch, epochs,fold,train_err/ train_batches, train_acc/train_batches*100,real_train_accuracy*100))
		# with open(validation_log, 'a') as text_file:
		# 	text_file.write("Epoch {} of {} fold: {} validation_loss: {:.6f} validation_accuracy: {:.2f} % ,real_validation_accuracy: {:.2f} %\n".format(epoch, epochs,fold, val_err/ val_batches,val_acc/val_batches*100,real_val_accuracy*100))

	if patience == True:
		#save the model for each fold
		with np.load(model_name) as f:
			# print f['arr_0'].shape
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			# print len(param_values)

		lasagne.layers.set_all_param_values(network, param_values)

		
	#training is over, compute the test loss
	# test_err = 0
	# test_acc = 0
	# test_batches = 0

	# test_pred = np.zeros([X_test.shape[0],num_classes])
	# test_voting_pred = np.zeros([X_test.shape[0],num_classes])
	# test_correct_pred = np.zeros(X_test.shape[0])
	# visit = np.zeros(X_test.shape[0])
	# num_test_replicates=division

	# for batch in PD.iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
	# 	# print " test_batches:\t{}".format(test_batches)
	# 	inputs, targets, indices = batch
	# 	test_error, test_accuracy, prediction= val_fn(inputs, targets)
	# 	test_err += test_error
	# 	test_acc += test_accuracy
	# 	test_batches += 1
	# 	for i in range(len(indices)):
	# 		visit[indices[i]/num_test_replicates] += 1
	# 		test_pred[indices[i]/num_test_replicates] += prediction[i]
	# 		test_voting_pred[indices[i]/num_test_replicates][np.argmax(prediction[i])] += 1
	# 		test_correct_pred[indices[i]/num_test_replicates] = targets[i]

	# # for v in visit:
	# # 	if v!=division:
	# # 		print '-----------Error occured in testing segmentation....................'
	# # 		print v
	# test_predicted_label = np.argmax(test_pred, axis=1)
	# real_test_accuracy = np.mean(np.equal(np.argmax(test_pred, axis=1),test_correct_pred))
	# voting_test_accuracy = np.mean(np.equal(np.argmax(test_voting_pred, axis=1),test_correct_pred))

	

	# print " final results:"
	# print " test loss:\t{:.6f}".format(test_err/test_batches)
	# print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
	# print " real test accuracy:\t{:.2f} %\n".format(real_test_accuracy*100)
	# # print " voting test accuracy:\t{:.2f} %\n".format(voting_test_accuracy*100)

	# with open('test_predicted_label.txt', 'w') as text_file:	
	# 	text_file.write('predicted\tground_truth\n')
	# 	for i in range(len(test_pred)):
	# 		# print int(test_predicted_label[i]),int(test_correct_pred[i])
	# 		# print my_map[int(test_predicted_label[i])],my_map[int(test_correct_pred[i])]
	# 		text_file.write(my_map[int(test_predicted_label[i])] + '\t' + my_map[int(test_correct_pred[i])] + '\n')

	test_err, real_test_acc, test_acc = iterate('test', X_test,y_test,num_replicates,batch_size,shuffle=False)

	# with open(test_log, 'a') as text_file:
	# 	text_file.write(" Epoch {} of {} fold: {} test loss: {:.6f} test accuracy: {:.2f} % ,real_test_accuracy: {:.2f} \n%".format(epoch, epochs,fold,test_err/ test_batches,test_acc/test_batches*100, real_test_accuracy*100))

	if patience==False:
		np.savez(model_name, *lasagne.layers.get_all_param_values(network))


# dataset_name = sys.argv[1]
# fold = int(sys.argv[2])
# division = int(sys.argv[3])
# patience = sys.argv[4]
# representation = sys.argv[5]
# augmentation = sys.argv[6]
# train()
def main():
    # parse command line
    my_parser = parser.opts_parser()
    args = my_parser.parse_args()

    # run
    train(**vars(args))


if __name__ == "__main__":
	main()
