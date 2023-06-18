import os
import subprocess


for file in os.listdir():
    if file.endswith('.tar'):
        d = file.split('.')[0]
        os.makedirs(d, exist_ok=True)
        subprocess.run(['tar', 'xf', file,'-C',d])
print('Done')