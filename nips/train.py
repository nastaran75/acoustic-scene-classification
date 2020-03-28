import PrepareDataset as PD
import theano
import lasagne
import theano.tensor as T
import DCASE2017_Network 
from time import gmtime, strftime
import os
import numpy as np

log_dir = 'log/'

def train():

	learning_rate = 0.002
	num_epoches = 100
	batch_size = 50
	for i in range(1,5):
		#tensor variables for inputs and targets
		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')

		network = DCASE2017_Network.build_cnn(input_var)

		#the final prediction of the network
		prediction = lasagne.layers.get_output(network)

		#define the loss
		loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		loss = loss.mean()

		#update the parameters based on SGD with nesterov momentum
		params = lasagne.layers.get_all_params(network, trainable = True)
		updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

		#loss for validation/testing, note that deterministic=true means we disable droput layers for test/eval
		test_prediction = lasagne.layers.get_output(network, deterministic = True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
		test_loss = test_loss.mean()

		#computing the accuracy
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var), dtype=theano.config.floatX)

		#perform the training
		train_fn = theano.function([input_var, target_var], loss, updates = updates)

		#test and eval (no updates)
		val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

		date = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
		train_log = log_dir + date+'_training_log.txt'
		validation_log = log_dir + date + '_validation_log.txt'
		test_log = log_dir + date + '_test_log.txt'

		#iterate over epoches
		print("Loading data...")
		X_train, y_train, X_val, y_val = PD.load_dataset(i)
		for epoch in range(num_epoches):
			# print " epoch:\t{}".format(epoch)
			#iterate over the whole training set in each epoch
			train_err = 0
			train_batches = 0
			for batch in PD.iterate_minibatches(X_train, y_train, batch_size, shuffle = True):
				print " train_batches:\t{}".format(train_batches)
				inputs, targets = batch
				train_err += train_fn(inputs, targets)
				train_batches += 1

			#iterate over the validation set
			val_err = 0
			val_acc = 0.0
			val_batches = 0
			for batch in PD.iterate_minibatches(X_val, y_val, batch_size, shuffle = False):
				# print " val_batches:\t{}".format(val_batches)
				inputs, targets = batch
				val_error, val_accuracy = val_fn(inputs, targets)
				val_err += val_error
				val_acc += val_accuracy
				val_batches += 1

			print " Epoch {} of {} ".format(epoch + 1, num_epoches)
			print " training loss:\t{:.6f}".format(train_err/ train_batches)
			print " validation loss:\t{:.6f}".format(val_err/ val_batches)
			print " validation accuracy:\t{:.2f} %".format(val_acc/val_batches*100)
			print "---------------------------------------------------------------------------------------"
			with open(train_log, 'a') as text_file:
				text_file.write("Epoch {} of {} fold: {} training_loss: {:.6f} \n".format(epoch, num_epoches,i,train_err/ train_batches))
			with open(validation_log, 'a') as text_file:
				text_file.write(" Epoch {} of {} fold: {} validation_loss: {:.6f} validation_accuracy: {:.2f} %\n".format(epoch, num_epoches,i, val_err/ val_batches,val_acc/val_batches*100))

		#training is over, compute the test loss
		test_err = 0
		test_acc = 0
		test_batches = 0

		X_test, y_test = PD.load_testset()
		for batch in PD.iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
			print " test_batches:\t{}".format(test_batches)
			inputs, targets = batch
			test_error, test_accuracy = val_fn(inputs, targets)
			test_err += test_error
			test_acc += test_accuracy
			test_batches += 1

		print "final results:"
		print " test loss:\t{:.6f}".format(test_err/test_batches)
		print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
		with open(test_log, 'a') as text_file:
			text_file.write(" Epoch {} of {} fold: {} test loss: {:.6f} test accuracy: {:.2f} \n%".format(epoch, num_epoches,i,test_err/ test_batches,test_acc/test_batches*100))


		#save the model for each fold
		np.savez('model_' + str(i) + '.npz', *lasagne.layers.get_all_param_values(network))

train()

