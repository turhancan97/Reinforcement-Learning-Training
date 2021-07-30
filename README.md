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
* *Homework*

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
* *Homework*

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
* *Homework*

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
* *Homework*

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
* *Homework*

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
* *Homework*

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
* *Homework*

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
	
There are five fundamental principle of Reinforcement Learning:
1. Principle #1: The input and output system
2. Principle #2: The reward
3. Principle #3: The AI environment
4. Principle #4: The Markov decision process
5. Principle #5: Training and inference

```python
# The 5 Core Principles Pseudocode

class Environment():

	def __init__(self):
		Initialize the game
	
	def get_observation(self):
		Return the state of the game 
	
	def get_reward(self, action):
		Return the reward obtained by playing this action
	
	def update(self, action):
		Update the environment based on the action specified

class AI():

	def __init__(self):
		Initialize the AI
	
	def train(self, state_of_the_game, reward):
		Train the AI based on the state of the game and the reward obtained

	def play_action(self, state_of_the_game):
		Play an action based on the state of the game

def markov_decision_process_training():
	env = Environment()
	ai = AI() 
	while True:
		state_of_the_game = env.get_observation()
		action = ai.play_action(state_of_the_game)
		reward = env.get_reward(action)
		ai.train(state_of_the_game, reward)
		env.update(action)

def markov_decision_process_inference():
	env = Environment()
	ai = AI() 
	while True:
		state_of_the_game = env.get_observation()
		action = ai.play_action(state_of_the_game)
		env.update(action)
```
- First AI Model
	> The Multi-Armed Bandit Problem and The Thompson Sampling Model

**[Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling "Multi-armed bandit")**, named after William R. Thompson, is a heuristic for choosing actions that addresses the exploration-exploitation dilemma in the [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit "Multi-armed bandit") problem. It consists of choosing the action that maximizes the expected reward with respect to a randomly drawn belief. _See_ Wikipedia,  Thompson sampling, https://en.wikipedia.org/wiki/Thompson_sampling
```python
# Thompson Sampling for Slot Machines

# Importing the libraries
import numpy as np

# Setting conversion rates and the number of samples
conversionRates = [0.15, 0.04, 0.13, 0.11, 0.05]
N = 10000
d = len(conversionRates)

# Creating the dataset
X = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversionRates[j]:
            X[i][j] = 1

# Making arrays to count our losses and wins
nPosReward = np.zeros(d)
nNegReward = np.zeros(d)

# Taking our best slot machine through beta distibution and updating its losses and wins
for i in range(N):
    selected = 0
    maxRandom = 0
    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j] + 1, nNegReward[j] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j
    if X[i][selected] == 1:
        nPosReward[selected] += 1
    else:
        nNegReward[selected] += 1

# Showing which slot machine is considered the best
nSelected = nPosReward + nNegReward 
for i in range(d):
    print('Machine number ' + str(i + 1) + ' was selected ' + str(nSelected[i]) + ' times')
print('Conclusion: Best machine is machine number ' + str(np.argmax(nSelected) + 1))
```
*Output:*
> Machine number 1 was selected 8131.0 times
> 
>Machine number 2 was selected 101.0 times
>
>Machine number 3 was selected 911.0 times
>
>Machine number 4 was selected 613.0 times
>
>Machine number 5 was selected 244.0 times
>
>Conclusion: Best machine is machine number 1

**Thompson Sampling vs. the Standard Model:**
```python
# Models comparison

import numpy as np
import pandas as pd

N = [200, 1000, 5000]
D = 20
convRanges = [(0., 0.1), (0., 0.3), (0., 0.5)]

results = list()
for n in N:
    for ranges in convRanges:
        results.append([])
        for d  in range(3, D + 1):
            p1 = 0
            p2 = 0

            for rounds in range(1000):
                
                conversionRates = list()
                for i in range(d):
                    conversionRates.append(np.random.uniform(low = ranges[0], high = ranges[1]))
                    
                X = np.zeros((n,d))
                for i in range(n):
                    for j in range(d):
                        if np.random.rand() < conversionRates[j]:
                            X[i][j] = 1
                
                nPosReward = np.zeros(d)
                nNegReward = np.zeros(d)
                
                for i in range(n):
                    selected = 0
                    maxRandom = 0
                    
                    for j in range(d):
                        randomBeta = np.random.beta(nPosReward[j] + 1, nNegReward[j] + 1)
                        if randomBeta > maxRandom:
                            maxRandom = randomBeta
                            selected = j
                        
                    if X[i][selected] == 1:
                        nPosReward[selected] += 1
                    else:
                        nNegReward[selected] += 1
                
                nSelected = nPosReward + nNegReward
                
                left = n - max(nSelected)
                
                countStandard = np.zeros(d)
                
                x = int(left / d)
                for i in range(x):
                    for j in range(d):
                        if X[i][j] == 1:
                            countStandard[j] += 1
                
                bestStandard = np.argmax(countStandard)
                bestReal = np.argmax(conversionRates)
                bestTS = np.argmax(nSelected)

                if bestTS == bestReal:
                    p1 += 1
                if bestStandard == bestReal:
                    p2 += 1
                
            print('N = ' + str(n) + ' d = ' + str(d) + ' range = ' + str(ranges) + ' | result Thompson Sampling = ' + str(p1) + ' result Standard solution = ' + str(p2))
            results.append([n, ranges, d, p1, p2])
                
df = pd.DataFrame(results)
df.to_excel('results.xlsx', sheet_name = 'Result', index = False)
```
*Output:*
> N = 200 d = 3 range = (0.0, 0.1) | result Thompson Sampling = 666 result Standard solution = 577
> 
>N = 200 d = 4 range = (0.0, 0.1) | result Thompson Sampling = 547 result Standard solution = 476
>
>N = 200 d = 5 range = (0.0, 0.1) | result Thompson Sampling = 483 result Standard solution = 415
>
>N = 200 d = 6 range = (0.0, 0.1) | result Thompson Sampling = 422 result Standard solution = 388
>
>N = 200 d = 7 range = (0.0, 0.1) | result Thompson Sampling = 373 result Standard solution = 348
>
>N = 200 d = 8 range = (0.0, 0.1) | result Thompson Sampling = 314 result Standard solution = 326
>
>N = 200 d = 9 range = (0.0, 0.1) | result Thompson Sampling = 320 result Standard solution = 265
>
>N = 200 d = 3 range = (0.0, 0.3) | result Thompson Sampling = 786 result Standard solution = 658
>
>N = 200 d = 4 range = (0.0, 0.3) | result Thompson Sampling = 714 result Standard solution = 623
>
>N = 200 d = 5 range = (0.0, 0.3) | result Thompson Sampling = 634 result Standard solution = 522
>
>N = 200 d = 6 range = (0.0, 0.3) | result Thompson Sampling = 594 result Standard solution = 495
>
>N = 200 d = 7 range = (0.0, 0.3) | result Thompson Sampling = 496 result Standard solution = 415
>
>N = 200 d = 8 range = (0.0, 0.3) | result Thompson Sampling = 486 result Standard solution = 373
>
>N = 200 d = 9 range = (0.0, 0.3) | result Thompson Sampling = 469 result Standard solution = 352
- AI for Sales and Advertising
	> Problem to Solve

**Click Here for the Application**
- Q-Learning
	> Q-Learning Pseudocode

**_Q_-learning** is a [model-free](https://en.wikipedia.org/wiki/Model-free_(reinforcement_learning) "Model-free (reinforcement learning)")  [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) algorithm to learn the value of an action in a particular state. It does not require a model of the environment (hence "model-free"), and it can handle problems with stochastic transitions and rewards without requiring adaptations. _See_ Wikipedia, Q-learning, https://en.wikipedia.org/wiki/Q-learning

```python
class Environment():

	def __init__(self):
		Initialize the Environment
	
	def get_random_state(self):
		Return a random possible state of the game
	
	def get_qvalue(self, random_state, action):
		Return the Q-value of this random_state, action couple	

	def update(self, action):
		Update the environment, reach the next state and return the Q-values of this new state

	def get_reward(self, random_state, action):
		Return the reward obtained by playing this action from this random possible state
	
	def calculate_TD(self, qvalue, next_state, reward, gamma):
		Return the calculated Temporal Difference using the equation: TD = reward + gamma*max(qvalues_next_state) - qvalue

	def update_qvalue(self, TD, qvalue, alpha):
		Update the qvalue specified as argument using the equation: qvalue = qvalue + alpha * TD

class AI():
	
	def __init__(self):
		Initialize the AI

	def play_action(self):
		Play a random action												##


env = Environment()
ai = AI()

Initialize gamma
Initialize alpha

while True:
	random_state = env.get_random_state()

	action = ai.play_action()
	
	qvalue = env.get_qvalue(random_state, action)

	next_state = env.update(action)

	reward = env.get_reward(random_state, action)

	TD = env.calculate_TD(qvalue, next_state, reward, gamma)

	env.update_qvalue(TD, qvalue, alpha)
```
- AI for Logistics
	> Robotics Application - Problem to solve

**Click Here for the Application**
- Deep Q-Learning
	> Deep Learning Theory and using with Q-Learning

Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

* In deep Q-learning, neural network to approximate the Q-value function is used. 		The state is given as the input and the Q-value of all possible actions is generated as the output.

*Predicting house price with deep learning* - **Application Time:** 

```python
# Predicting House Prices 

# Importing the libraries
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')

# Getting separately the features and the targets
X = dataset.iloc[:, 3:].values
X = X[:, np.r_[0:13,14:18]]
y = dataset.iloc[:, 2].values

# Splitting the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scaling the features
xscaler = MinMaxScaler(feature_range = (0,1))
X_train = xscaler.fit_transform(X_train)
X_test = xscaler.transform(X_test)

# Scaling the target
yscaler = MinMaxScaler(feature_range = (0,1))
y_train = yscaler.fit_transform(y_train.reshape(-1,1))
y_test = yscaler.transform(y_test.reshape(-1,1))

# Building the Artificial Neural Network
model = Sequential()
model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
model.compile(optimizer = Adam(lr = 0.001), loss = 'mse', metrics = ['mean_absolute_error'])

# Training the Artificial Neural Network
model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data = (X_test, y_test))

# Making predictions on the test set while reversing the scaling
y_test = yscaler.inverse_transform(y_test)
prediction = yscaler.inverse_transform(model.predict(X_test))

# Computing the error rate
error = abs(prediction - y_test)/y_test
print(np.mean(error))
```
- AI for Autonomous Vehicle
	> Self Driving Car Application
	
**Click Here for the Application**
- AI for Business
	> Business Application
	
**Click Here for the Application**
- Deep Convolutional Q-Learning
	> Basics of CNN
	
*Cat and Dog Classification* - **Application Time:** 

In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of artificial neural network, most commonly applied to analyze visual imagery.
```python
# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Optional Step - Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Step 5 - Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Training the CNN

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

# Part 3 - Making a new prediction on a single image

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
```
- AI for Games
	> Game Application
	
**Click Here for the Application**
