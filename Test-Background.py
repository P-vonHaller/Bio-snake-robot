import time
import sys
for i in range(1000):
    print('%d ht Time...' %i)
    sys.stdout.flush()
    f = open('Background.txt', 'a')
    f.write('Something Something %d ht Time...\n' %i)
    f.close()
    time.sleep(3)
