import numpy as np
import time
import random
from random import random, randint

# UI Libraries
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Model
from model import DQN

# Config
Config.set('input', 'mouse', 'multitouch_on_demand')

'''
Config

Initializing the last_x and last_y i.e, the cordinates
used to keep the point in memory when we draw the sand in the map
'''
last_x = 0
last_y = 0
n_points = 0
length = 0

brain = DQN(5, 3, 0.9) # 5 states/inputs, 3 actions(left, right, straight), 0.9 discount factor
action_rotation = [0, 20, -20] # 0-> no change(straight), 20 -> go right, -20 -> go left
last_reward = 0 # Penalize the car if it goes to sand else +ve reward
scores = [] # Plot reward vs Episode

'''
Initialize a Map

Sand will be pixels in the map i.e, an array 1 if sand else if no sand then 0
At beginning there will be no sand so values will be zero
Goal X is destination or say goal the car want to accomplish, 
suppose there is some track and goal is to reach the tp left corner so car should go that way
Goal Y is like back to home, once Goal X is achived then Goal Y will be triggred
We will make the road as complicated as possible to make sure it reach both goals
we have to avoid walls and the grass/sand

'''
first_update = True

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((long, larg))
    goal_x = 20
    goal_y = larg-20
    first_update = False

'''
Initialize the last distance

It gives the current distance of the car so 0 for now
'''
last_distance = 0