import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath + '/../'))
sys.path.insert(1, os.path.join(myPath + '/../src/'))
