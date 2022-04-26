import argparse
from time import sleep
from tkinter.tix import INTEGER

parser = argparse.ArgumentParser()
parser.add_argument("num1", help="first number")
# parser.add_argument("num2", help="second number")
# parser.add_argument("operation", help="choose operation")
args = parser.parse_args()


# f= open("same_words.txt", "a")
# f.write("My name is Harshit \n")
# f.close()

print(args.num1)
for i in range(0, 100000):
    print(i)


# for i in range(0, 10):
#     print("exit here")