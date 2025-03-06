import numpy as np
import time
import random
from random import random, randint
import matplotlib.pyplot as plt

# UI Libraries
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.lang import Builder

Builder.load_file('car.kv')

# Model
from model import DQN

# Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

'''
Config

Initializing the last_x and last_y i.e, the cordinates
used to keep the point in memory when we draw the sand in the map
'''
last_x = 0
last_y = 0
n_points = 0
length = 0
longg = 0  # Initialize with default values
larg = 0   # Initialize with default values

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
sand = None  # Initialize as None and properly create it in init()

def init():
    global sand
    global x_destination
    global y_home
    global first_update
    sand = np.zeros((longg, larg))
    x_destination = 20
    y_home = larg-20
    first_update = False

'''
Initialize the last distance

It gives the current distance of the car so 0 for now
'''
last_distance = 0

'''
Car class

So to build it we need to understand some ethics how it works and what all info we 
need to be aware of for example angle of car, velocity of car, rotation of car, sensors etc
So there will be 3 sensor in our car, Sensor 1 will be check is there any object infront of car
Sensor 2, any object on left
Sensor 3, any object on right and then from these sensors we will recieve the signal
Signal 1 is the signal recieved from Sensor 1
Signal 2 is the signal recieved from Sensor 2
Signal 3 is the signal recieved from Sensor 3
this is calculated using density function
signal 1 is the density of sand around sensor 1: we take squares of each of sensor
which is 200x200, and for each of the squares we divide number of ones in the square by
all number of cell in square, which 20x20 = 400 that gives density, because the ones corresponds to sand
we do it for all sensor 
'''
class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        #allow to go left, right, straight
        ##pos is last position, position will be updated in the direction of velocity vector
        self.pos = Vector(*self.velocity) + self.pos 
        #how we gonna rotate the car going to left or right
        self.rotation = rotation
        #angle between x axis and the axis of the direction of the car
        self.angle = self.angle + self.rotation
        #once the car is moved then we have to update the sensor and the signal
        #so if car is rotated sensor will also get rotates, so we updated using rotate fn. and to which we add new pos
        #30 is the difference between sensor and car
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        
        # Fix attribute names to match property names
        self.sensor1_x, self.sensor1_y = self.sensor1
        self.sensor2_x, self.sensor2_y = self.sensor2
        self.sensor3_x, self.sensor3_y = self.sensor3
        
        #once sensor updated its time for signal
        #here we get x1 sensor and we take all the cell values from +10 to -10 then we do same for y cordinates from -10 to +10
        #we get square of 20x20 pixels sorrounding the sensor, and inside the square we sum all the ones, so 20x20 is 400 cells so thats
        # we divivded it by 400 to get the density of ones inside the square, thats to detect sand
        
        # Avoid index errors by ensuring indices are within bounds
        x1_min = max(0, int(self.sensor1_x) - 10)
        x1_max = min(longg - 1, int(self.sensor1_x) + 10)
        y1_min = max(0, int(self.sensor1_y) - 10)
        y1_max = min(larg - 1, int(self.sensor1_y) + 10)
        
        x2_min = max(0, int(self.sensor2_x) - 10)
        x2_max = min(longg - 1, int(self.sensor2_x) + 10)
        y2_min = max(0, int(self.sensor2_y) - 10)
        y2_max = min(larg - 1, int(self.sensor2_y) + 10)
        
        x3_min = max(0, int(self.sensor3_x) - 10)
        x3_max = min(longg - 1, int(self.sensor3_x) + 10)
        y3_min = max(0, int(self.sensor3_y) - 10)
        y3_max = min(larg - 1, int(self.sensor3_y) + 10)
        
        # Calculate signal density properly
        area1 = max(1, (x1_max - x1_min) * (y1_max - y1_min))  # Avoid division by zero
        area2 = max(1, (x2_max - x2_min) * (y2_max - y2_min))
        area3 = max(1, (x3_max - x3_min) * (y3_max - y3_min))
        
        self.signal1 = int(np.sum(sand[x1_min:x1_max, y1_min:y1_max])) / area1
        self.signal2 = int(np.sum(sand[x2_min:x2_max, y2_min:y2_max])) / area2
        self.signal3 = int(np.sum(sand[x3_min:x3_max, y3_min:y3_max])) / area3
        
        #this below line is for rewarding bad if it reaches the one of the edges in map
        ##########right edge################left edge#################top edge##############bottom edge############
        if self.sensor1_x > longg - 10 or self.sensor1_x < 10 or self.sensor1_y > larg - 10 or self.sensor1_y < 10:
            #it will stop the car, worst value it can get is 1 i.e, bad reward
            self.signal1 = 1
        if self.sensor2_x > longg - 10 or self.sensor2_x < 10 or self.sensor2_y > larg - 10 or self.sensor2_y < 10:
            self.signal2 = 1
        if self.sensor3_x > longg - 10 or self.sensor3_x < 10 or self.sensor3_y > larg - 10 or self.sensor3_y < 10:
            self.signal3 = 1


class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self): # Initialize position and vector
        self.car.center = self.center
        self.car.velocity = Vector(6, 0) # x = 6 and y = 0
    
    def update(self, dt):
        global brain # store knowledge of the environment and generate action
        global last_reward # reward of last action taken by agent
        global scores # score of agent over time
        global last_distance # distance of agent (car) from the goal in the previous time step
        global x_destination # x coordinate of goal
        global y_home # y coordinate of goal
        global longg # width of the screen (playground)
        global larg # height of the screen (playground)

        longg = self.width
        larg = self.height
        if first_update:
            init()
        
        # xx and yy compute the distance between car and the goal
        xx = x_destination - self.car.x
        yy = y_home - self.car.y

        # orientation of car with respect to goal is the calculated by computing the angle between car velocity and the vector pointing towards the goal
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180
        
        #creating last signal, since its obvious we have to take from all 3
        #adding orientation wrt goal beucase: if its heading towards goal then orientation will be 0,
        # if it goes slightly towards right then orientation will be close to 45 degree
        # or left then -45 degree, adding -orientation in the car means stablizing the exploration i.e, both direction not just one
        #these 5 input will go to the agent as encoded vector
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        #and it return actions
        action = brain.update(last_reward, last_signal)
        #update mean score
        scores.append(brain.score())
        #we update rotation based on action
        rotation = action_rotation[action]
        #move the car based on rotation
        self.car.move(rotation)
        #we update the distance
        distance = np.sqrt((self.car.x - x_destination)**2 + (self.car.y - y_home)**2)
        #position updation
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        #if car in sand, reduce the velocity also get bad reward
        # Check if car position is within bounds
        car_x = max(0, min(longg - 1, int(self.car.x)))
        car_y = max(0, min(larg - 1, int(self.car.y)))
        
        if sand[car_x, car_y] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            #if closer to goal get good reward, if it deviates from goal get slight bad reward
            #6 because it should keep usual speed
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        #last condition for rewards
        #if car comes to close to edges it get -1 reward
        if self.car.x < 10: # left edge
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10: # right edge
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10: # top edge
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10: # bottom edge
            self.car.y = self.height - 10
            last_reward = -1
        #once reached goal
        #we update the x cordinate of goal as well as y cordinates
        #and then we update distance from car to goal
        if distance < 100:
            x_destination = self.width - x_destination
            y_home = self.height - y_home
        last_distance = distance

#Adding the paint tools
class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10
            touch.ud['line'] = Line(points = (touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            
            # Make sure the indices are within bounds
            if 0 <= int(touch.x) < longg and 0 <= int(touch.y) < larg:
                sand[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            with self.canvas:
                touch.ud['line'].points += [touch.x, touch.y]
                x = int(touch.x)
                y = int(touch.y)
                length += np.sqrt(max(1, (x-last_x)**2 + (y-last_y)**2))  # Avoid sqrt of negative
                n_points += 1
                density = n_points/(length) if length > 0 else 1  # Avoid division by zero
                touch.ud['line'].width = int(20*density+1)
            
                # Fix: Actually update the sand array
                # Make sure the indices are within bounds
                # x_min = max(0, int(touch.x) - 10)
                # x_max = min(longg, int(touch.x) + 10)
                # y_min = max(0, int(touch.y) - 10)
                # y_max = min(larg, int(touch.y) + 10)
                
                # if x_min < x_max and y_min < y_max:  # Ensure valid slice
                #     sand[x_min:x_max, y_min:y_max] = 1
            
                for ix in range(max(0, x-10), min(longg, x+10)):
                    for iy in range(max(0, y-10), min(larg, y+10)):
                        sand[ix, iy] = 1
                    
                last_x = x
                last_y = y

#adding clear, save and load button
class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        
        # Fix: Improve button positioning and sizing
        button_height = 50
        button_width = 100
        
        # Set button positions and sizes correctly
        clearbtn = Button(text='clear', size=(button_width, button_height), 
                          pos=(0, 0), size_hint=(None, None))
        savebtn = Button(text='save', size=(button_width, button_height),
                        pos=(button_width, 0), size_hint=(None, None))
        loadbtn = Button(text='load', size=(button_width, button_height),
                        pos=(2 * button_width, 0), size_hint=(None, None))
        
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        
        # Add widgets in the right order
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        
        return parent

    def clear_canvas(self, obj):
        global sand
        global longg, larg
        self.painter.canvas.clear()
        # Make sure longg and larg are properly initialized before using them
        if longg > 0 and larg > 0:
            sand = np.zeros((longg, larg))
        else:
            print("Warning: Screen dimensions are not properly initialized")
            
    #to save the model and re use it later
    def save(self, obj):
        print('Saving model')
        brain.save()
        plt.plot(scores)
        plt.show()
        
    def load(self, obj):
        print('Loading the last saved model')
        brain.load()
        
if __name__ == "__main__":
    CarApp().run()