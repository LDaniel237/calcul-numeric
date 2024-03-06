import random
import math

def findMachinePrecision():
  u = 1;

  while (1 + u != 1):
    u = u / 10;

  return u * 10;

def isAdditionAssociative():
  u = findMachinePrecision()
  x = 1.0
  y = u / 10
  z = u / 10
  return (x + y) + z == x + (y + z)

def isMultiplicationAssociative():
  x = random.random()
  y = random.random()
  z = random.random()
  while (x * y) * z == x * (y * z):
    x = random.random()
    y = random.random()
    z = random.random()
    
  return {
    "first": x,
    "second": y,
    "third": z,
  }

def t4(a):
  denominator = 105 * a - 10 * a ** 3
  nominator = 105 - 45 * a ** 2 + a ** 4
  return nominator / denominator

def t5(a):
  denominator = 945 * a - 105 * a ** 3 + a ** 5
  nominator = 945 - 420 * a ** 2 + 15 * a ** 4
  return nominator / denominator

def t6(a):
  denominator = 10395 * a - 1260 * a ** 3 + 21 * a ** 5
  nominator = 10395 - 4725 * a ** 2 + 210 * a ** 4 - a ** 6
  return nominator / denominator

def t7(a):
  denominator = 135135 * a - 17325 * a ** 3 + 378 * a ** 5 - a ** 7
  nominator = 135135 -  62370 * a ** 2 + 3150 * a ** 4 - 28 * a ** 6
  return nominator / denominator

def t8(a):
  denominator = 2027025 * a - 270270 * a ** 3 + 6930 * a ** 5 - 36 * a ** 7
  nominator = 2027025 -  945945 * a ** 2 + 51975 * a ** 4 - 630 * a ** 6 + a ** 8
  return nominator / denominator

def t9(a):
  denominator = 34459425 * a - 4729725 * a ** 3 + 135135 * a ** 5 - 990 * a ** 7 + a ** 9
  nominator = 34459425 - 16216200 * a ** 2 + 945945 * a ** 4 - 13860 * a ** 6 + 45 * a ** 8
  return nominator / denominator

def buildFunctionsHierarchy():
  randomNumbers = [random.uniform(-math.pi / 2, math.pi / 2) for i in range(10000)]

  t4Values = [t4(x) for x in randomNumbers]
  t5Values = [t5(x) for x in randomNumbers]
  t6Values = [t6(x) for x in randomNumbers]
  t7Values = [t7(x) for x in randomNumbers]
  t8Values = [t8(x) for x in randomNumbers]
  t9Values = [t9(x) for x in randomNumbers]

  tanValues = [math.tan(x) for x in randomNumbers]

  t4Differences = [abs(t4Values[i] - tanValues[i]) for i in range(10000)]
  t5Differences = [abs(t5Values[i] - tanValues[i]) for i in range(10000)]
  t6Differences = [abs(t6Values[i] - tanValues[i]) for i in range(10000)]
  t7Differences = [abs(t7Values[i] - tanValues[i]) for i in range(10000)]
  t8Differences = [abs(t8Values[i] - tanValues[i]) for i in range(10000)]
  t9Differences = [abs(t9Values[i] - tanValues[i]) for i in range(10000)]
  
  t4Count = 0
  t5Count = 0
  t6Count = 0
  t7Count = 0
  t8Count = 0
  t9Count = 0

  for i in range(10000):
    differences = [
      {
        "name": "t4",
        "difference": t4Differences[i]
      },
      {
        "name": "t5",
        "difference": t5Differences[i]
      },
      {
        "name": "t6",
        "difference": t6Differences[i]
      },
      {
        "name": "t7",
        "difference": t7Differences[i]
      },
      {
        "name": "t8",
        "difference": t8Differences[i]
      },
      {
        "name": "t9",
        "difference": t9Differences[i]
      }
    ]

    differences.sort(key=lambda x: x["difference"])
    if (differences[0]["name"] == "t4"):
      t4Count += 1
    elif (differences[0]["name"] == "t5"):
      t5Count += 1
    elif (differences[0]["name"] == "t6"):
      t6Count += 1
    elif (differences[0]["name"] == "t7"):
      t7Count += 1
    elif (differences[0]["name"] == "t8"):
      t8Count += 1
    elif (differences[0]["name"] == "t9"):
      t9Count += 1

    print("For x = " + str(randomNumbers[i]) + " the closest functions to tan are: ")
    for j in range(3):
      print(differences[j])

  results = [
    {
      "name": "t4",
      "count": t4Count
    },
    {
      "name": "t5",
      "count": t5Count
    },
    {
      "name": "t6",
      "count": t6Count
    },
    {
      "name": "t7",
      "count": t7Count
    },
    {
      "name": "t8",
      "count": t8Count
    },
    {
      "name": "t9",
      "count": t9Count
    }
  ];

  results.sort(key=lambda x: x["count"], reverse=True)

  return results[:3]

def ex1():
  print(findMachinePrecision())

def ex2():
  print(isAdditionAssociative())
  print(isMultiplicationAssociative())

def ex3():
  print(buildFunctionsHierarchy())

def main():
  number = int(input('Enter the number of the exercise: '))
  if (number == 1):
    ex1()
  elif (number == 2):
    ex2()
  elif (number == 3):
    ex3()
  else:
    print('Invalid number')

main()

