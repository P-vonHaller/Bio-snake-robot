import matplotlib.pyplot as plt

LOSS = '/root/Bio-snake-robot/Pytorch-Deeplab/20000Default.log'
ACC = '/root/Bio-snake-robot/Pytorch-Deeplab/20000DefaultAcc.log'

#ACC = 'Atest.txt'
#LOSS = 'LossTest.txt'

l = open(LOSS, 'r')
xl = [0]
yl = [0]
for line in l.readlines():
	if line[0] == 's':
		break
	if line[0] != 't':
		xl.append(int(line.split(';')[0]))
		yl.append(int(line.split(';')[1]))
l.close()

a = open(ACC, 'r')
xa = [0]
ya = [0]
for line in a.readlines():
	xa.append(int(line.split(';')[0]))
	ya.append(int(line.split(';')[1]))
a.close()


fig, ax = plt.subplots()

ax.plot(xl, yl, 'r')
ax.plot(xa, ya, 'b')

ax.set(xlabel='Number Steps', ylabel='Accuracy(red)/Loss(blue)',title='Loss and Accuracy')

#plt.show()
fig.savefig('some.png')
