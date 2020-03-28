import matplotlib.pyplot as plt
import numpy as np

num_epoches = 100
num_folds = 4


valid_loss = np.empty(num_epoches*num_folds)
valid_acc = np.empty(num_epoches*num_folds)
train_loss = np.empty(num_epoches*num_folds)

counter = 0

validation_log = open('log/2018-07-25_11:51:12_validation_log.txt') 
train_log = open('log/2018-07-25_11:51:12_training_log.txt')
validation_lines = validation_log.readlines()
train_lines = train_log.readlines()
for line_num in range(len(validation_lines)):
    token_val = validation_lines[line_num].split()
    token_train = train_lines[line_num].split()
    for i in range(len(token_val)):
        if token_val[i]=='loss:':
            valid_loss[counter] = token_val[i+1]
            break
    for i in range(len(token_val)):
        if token_val[i]=='accuracy:':
            valid_acc[counter] = token_val[i+1]
            break
    for i in range(len(token_train)):
        if token_train[i] == 'loss:':
            train_loss[counter] = token_train[i+1]
            break
    print valid_loss[counter],train_loss[counter]
    counter += 1


plt.gca().cla()
# plt.plot(train_loss, label="train_loss") 
# plt.plot(valid_loss, label="val_loss")
plt.plot(valid_acc, label="val_acc")


plt.legend()
plt.draw()
plt.show()