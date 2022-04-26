import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("num1", help="first number")
# parser.add_argument("num2", help="second number")
# parser.add_argument("operation", help="choose operation")
args = parser.parse_args()


print(args.num1)
for i in range(0, 2000):
    print("hello")