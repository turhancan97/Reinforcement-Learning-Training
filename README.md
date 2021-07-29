# Welcome to AI Crash Course!

## Introduction

In this repo, you can find the codes of the **AI CRASH COURSE** book written by **Hadelin de PONTEVES**. Some codes are directly taken from the book and some of them has been changed by me. 

## Topics
Topics covered in this book are as follows:
- Python Fundamentals
	> Quick Start for Python

**Displaying Text**
```python
# Introduction to Displaying Text
# Displaying text
print('Hello world!')
print("Hello Python!!")
print("Hello AI!!!")
```
*Homework*
Using only one print() method, try to display two or more lines.
```python
# Exercise for Displaying Text: Homework Solution
print('First line\nSecond line')    # We only have to write '\n' in the middle of the
                                    # text we want to display. This will separate it into 
                                    # two consecutive lines.
print(15*"-")
print("Turhan\nCan\nKargin")
```
**Variables and operations**
```python
# Introduction to Variables
x = 2
x = x + 5   #x += 5
x = x - 3   #x -= 3
x = x * 2.5 #x *= 2.5
x = x / 3   #X /= 3
y = 3
x += y
print(x)
X = 10
y = 3
z = x%y
print (z)
```
*Homework*
Try to find a way to raise one number to the power of another.
**Hint:** Try using the pow() built-in function for Python.
```python
# Exercise for Variables: Homework Solution

a = 3               # we will raise this number to a power
b = 4               # this is the power to which we will raise "a"
power = pow(a, b)   # pow() function lets us raise "a" to the "b" power
print(power)        # we display the calculated power
```
**Lists and arrays**
```python
# Introduction to Lists and Arrays
L1 = list()
L2 = []
L3 = [3,4,1,6,7,5]
L4 = [[2, 9, -5], [-1, 0, 4], [3, 1, 2]]

import numpy as np
nparray = np.zeros((5,5))
```
*Homework*
Try to find the mean of all the numbers in the L4 list. There are multiple solutions.
**Hint:** The simplest solution makes use of the NumPy library. Check out some
of its functions here: https://docs.scipy.org/doc/numpy/reference/
```python
# Exercise for Lists and Arrays: Homework Solution

import numpy as np                          # we import the Numpy library and give it abbreviation "np"

L4 = [[2, 9, -5], [-1, 0, 4], [3, 1, 2]]    # this is once again our L4 list

mean = np.mean(L4)                          # this method from Numpy class let's us easily calculate the mean of an array/list

print(mean)                                 # we display the mean calculated by function above
```
**if statements and conditions**
```python
# Introduction to If Statements
a = 5
if a > 0:
    print('a is greater than 0')
elif a == 0:
    print('a is equal to 0')
else:
    print('a is lower than 0')
```
*Homework*
Build a condition that will check if a number is divisible by 3 or not.
**Hint:** You can use a mathematical expression called modulo, which when used,
returns the remainder from the division between two numbers. In Python, modulo
is represented by %. For example:
5 % 3 = 2
71 % 5 = 1
```python
# Exercise for If Statements: Homework Solution

a = 12                                          # we will check the divisibility of this number by 3

if a % 3 == 0:                                  # we check if a is divisible by 3
    print(str(a) + ' is divisible by 3')        # if it is we display that this number is divisible 
else:                                           # we enter this condtition, if the number is not divisible by 3
    print(str(a) + ' is not divisible by 3')    # then we display that indeed this number is not divisible by 3

# NOTE: str() function lets us display a variable followed by a text, it changes this integer number "a" to a text
```
**for and while loops**
```python
# Introduction to For and While Loops
for i in range(1, 20):
    print(i)
L3 = [3,4,1,6,7,5]
for element in L3:
    print(element)
stop = False
i = 0
###############################################
while stop == False:  # alternatively it can be "while not stop:"
    i += 1
    print(i)
    if i >= 19:
        stop = True
L4 = [[2, 9, -5], [-1, 0, 4], [3, 1, 2]]
for row in L4:
    for element in row:
        print(element)
```
*Homework*
Build both for and while loops that can calculate the factorial of a positive integer
variable.
**Hint:** Factorial is a mathematical function that returns the product of all positive
integers lower or equal to the argument of this function. This is the equation:
f(n) = n * (n – 1) * (n – 2) *...* 1
Where:
* f(n) – the factorial function
* n – the integer in question, the factorial of which we are searching for
This function is represented by ! in mathematics, for example:
5! = 5 * 4 * 3 * 2 * 1 = 120
4! = 4 * 3 * 2 * 1 = 24
```python
# Exercise for For and While Loops: Homework Solution
n = 5                       # the number factorial of which we are searching for

# for loop approach
factorial = 1               # this will be out factorial, we give it value 1
for i in range(1, n + 1):   # we iterate through every positive integer lower or equal to n, remember, upper bounds are excluded in Python
    factorial *= i          # we multiply factorial by i, this is why we initialized factorial to 1, if we set it to 0, then the product would be equal to 0
print(factorial)            # we display the product, factorial

# while loop approach
factorial = 1               # this will be out factorial, we give it value 1
i = 1                       # we create a variable i, which as previously, will count the number of iterations
while i <= n:               # initializing "while" loop, that will work as long as i is less or equal to n
    factorial *= i          # once again, we multiply our factorial by i, also the reason why factorial and i were set to 1 not 0
    i += 1                  # we increase i by one so that our loop will finally end
print(factorial)            # we display the factorial
```
**Functions**
```python
# Introduction to Functions
def division(a, b):
    result = a / b
    return result
d = division(3, 5)
print(d)
```
*Homework*
Build a function to calculate the distance between two points on an x,y plane:
one with coordinates x1 and y1, and the other with coordinates x2 and y2.
```python
# Exercise for Functions: Homework Solution

def distance(x1, y1, x2, y2):                       # we create a new function "distance" that takes coordinates of both points as arguments
    d = pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5) # we calculate the distance between two points using the formula provided in the hint ("pow" function is power)
    return d                                        # this line means that our function will return the distance we calculated before

dist1 = distance(0, 0, 3, 4)                         # we call our function while inputting 4 required arguments, that are coordinates
print(dist1)                                         # we display the calculated distance
dist2 = distance(1, 1, 7, 9)                         
print(dist2)   
```
**Classes and objects**
```python
# Introduction to classes
class Bot():
    
    def __init__(self, posx, posy):
        self.posx = posx
        self.posy = posy
    
    def move(self, speedx, speedy):
        self.posx += speedx
        self.posy += speedy
    
bot = Bot(3, 4)
bot.move(2, -1)
print(bot.posx, bot.posy)
```
*Homework*
Your final challenge will be to build a very simple car class. As arguments, a car
object should take the maximum velocity at which the car can move (unit in m/s), as
well as the acceleration at which the car is accelerating (unit in m/s2). I also challenge
you to build a method that will calculate the time it will take for the car to accelerate
from the current speed to the maximum speed, knowing the acceleration (use the
current speed as the argument of this method).
```python
# Exercise for Classes: Homework Solution

class Car():                                                    # we initialize a class called "Car"
    
    def __init__(self, topSpeed, acc):                          # __init__ method, called when an object of this class is created
                                                                # takes maximum velocity and accelearation as arguments
        self.topSpeed = topSpeed                                # we create a variable called "topSpeed" associated with this class/object by "self"
        self.acceleration = acc                                 # variable "acceleration" also associated with this class/objct by "self"
    
    def calcTime(self, currentSpeed):                           # we create a new method that will calculate the time required for the car to accelerate to top speed 
        t = (self.topSpeed - currentSpeed) / self.acceleration  # we calculate this time using the equation provided in the hint
        return t                                                # this method has to return the calculated time, therefore we write "return"


car = Car(75, 3.5)                                              # we create an object of "Car" class that we call "car", remember that
                                                                # we need to input two arguments: top speed and acceleration
time = car.calcTime(30)                                         # we calculate the time required to accelerate from 30 m/s to 75 m/s using the "calcTime" method
print(time)                                                     # and in the end we can finally display this time
```
- Reinforcement Learning 
	> Basics of Reinforcement Learning
- First AI Model
	> The Multi-Armed Bandit Problem and The Thompson Sampling Model
- AI for Sales and Advertising
	> Thompson Sampling vs. Random Selection
- Q-Learning
	> Q-Learning Pseudocode
- AI for Logistics
	> Robotics Application
- Deep Q-Learning
	> Deep Learning Theory and using with Q-Learning
- AI for Autonomous Vehicle
	> Self Driving Car Application
- AI for Business
	> Business Application
- Deep Convolutional Q-Learning
	> Basics of CNN
- AI for Games
	> Game Application
