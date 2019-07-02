import time
f = open('Background.txt', 'w')
for i in range(1000):
	print('%d ht Time...' %i)
	f.write('%d ht Time...\n' %i)
	time.sleep(3)