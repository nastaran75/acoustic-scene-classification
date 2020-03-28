import matplotlib.pyplot as plt
import numpy as np

num_epoches = 200
num_folds = 1


valid_loss = np.empty(num_epoches*num_folds)
valid_acc = np.empty(num_epoches*num_folds)
train_loss = np.empty(num_epoches*num_folds)
train_acc = np.empty(num_epoches*num_folds)

counter = 0

validation_log = open('log/final_2018-10-19_12:23:22fold_2_validation_log.txt') 
train_log = open('log/final_2018-10-19_12:23:22fold_2_validation_log.txt')
validation_lines = validation_log.readlines()
train_lines = train_log.readlines()
for line_num in range(len(validation_lines)):
    token_val = validation_lines[line_num].split()
    token_train = train_lines[line_num].split()
    for i in range(len(token_val)):
        if token_val[i]=='validation_loss:':
            valid_loss[counter] = token_val[i+1]
            break
    for i in range(len(token_val)):
        if token_val[i]=='real_validation_accuracy:':
            valid_acc[counter] = token_val[i+1]
            break
    for i in range(len(token_train)):
        if token_train[i] == 'training_loss:':
            train_loss[counter] = token_train[i+1]
            break
    for i in range(len(token_train)):
        if token_train[i]=='real_train_accuracy:':
            train_acc[counter] = token_train[i+1]
            break
    print valid_loss[counter],train_loss[counter],valid_acc[counter],train_acc[counter]
    counter += 1

fig = plt.figure()

plt.gca().cla()
plt.plot(train_loss, label="final_train_loss_fold2") 
plt.plot(valid_loss, label="final_val_loss_fold2")



plt.legend()
plt.draw()
plt.show()
# fig.savefig('../plot/fold4_loss.png')
# plt.figure()
plt.gca().cla()
plt.plot(valid_acc, label="val_acc_fold2")
# fig.savefig('../plot/fold4_acc.png')
plt.show()