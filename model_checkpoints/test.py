import argparse

tank_to_fish = {
    "tank_a": "shark, tuna, herring",
    "tank_b": "cod, flounder",
}

parser = argparse.ArgumentParser()
parser.add_argument("num1", help="first number")
parser.add_argument("num2", help="second number")
parser.add_argument("operation", help="choose operation")
args = parser.parse_args()

print(args.num1)
print(args.num2)
print(args.operation)

# count = 0
# for i in range(0, 1000000000):
#     count += i
