import atexit
import time

names = ['Geeks', 'for', 'Geeks']

def write():
    with open("temp.txt","w") as f:
        start_time = time.time()
        f.write(f"atexit called time {start_time}")
def hello(name):
    print (name)

# Using register()
atexit.register(hello, "atexit called")
atexit.register(write)

count =0
while(True):
    time.sleep(10)
    # if count==2:
        # raise Exception("dummy exception")
    count+=1
