import subprocess
import threading
from time import sleep
import numpy as np
from threading import Thread, Event
import sys
import numpy as np
import os

output = subprocess.Popen(["free", "-m"])
(out, err) = output.communicate()
print(out)