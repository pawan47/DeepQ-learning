import gym
import numpy as np
import keras
import random
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#from gym.wrappers import Monitor
#from gym import wrappers
import cv2
max_ep=200
train = True


class dqnagent():
    def __init__(self, lr = 0.00025, ob_size = (84,84,4), action_size = 4, env = 'BreakoutNoFrameskip-v0'):
        self.state_size = (84,84,4)
        self.action_size = 4
        # build model to extimate q value
        self.model = self._build_model(lr)
        # build target model
        self.model_t = self._build_model(lr)
        self.replay_memory = deque(maxlen = 100000)  # experience replay_memory to store value
        self.reward_memory = deque(maxlen = 100)
        self.env = gym.make(env)
        #self.env = Monitor(env, "/tmp",force = True)
        self.ep_start = 1
        self.ep_stop = .1
        self.ep = 1
        self.ep_decay = (1-.1)/500000
        self.batch_size = 32
        self.gamma = 0.99
        self.t = 0
        # make target and main model same first then after end of every episode we will update it
        self.update_target_model()

    def add_memory(self,s,a,r,d,s2):
        # adding experience replay memory
        self.replay_memory.append((s, a, r, d, s2))


    def choose_action(self,s):
        ran = np.random.random()

        # you can use non linear decay rate but we will use linear decay for to get good result ####---
        self.t +=1
        #self.ep = self.ep_stop + (self.ep_start - self.ep_stop)*np.exp(-self.ep_decay*self.t_)
        self.ep = np.max(self.ep_stop, self.ep_start - self.t*self.ep_decay)
        if self.ep >= ran :
            #self.ep -= self.ep_decay
            return self.env.action_space.sample()
        else:
            a = self.model.predict(s)
            return np.argmax(a[0])

    def learn(self):
        st_ = np.zeros((self.batch_size,84,84,4))
        st_2 = np.zeros((self.batch_size,84,84,4))
        out = np.zeros((self.batch_size,4))
        batch = random.sample(self.replay_memory, self.batch_size)
        i=0
        for s, a , r, d, s2 in batch:
            st_[i:i+1] = s
            st_2[i:i+1] = s2
            target = r
            if d == False:
                target = r + self.gamma * np.amax( self.model_t.predict(s2)[0] )
            out[i] = self.model.predict(s)
            out[i][a] = target
            i = i +1

        self.model.fit(st_,out,epochs=1,verbose=0)

    def _build_model(self,lr):
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        shape_image=(84,84,4)
        model = Sequential()
        model.add(keras.layers.Lambda(lambda x: x / 255.0,input_shape = shape_image))
        model.add(Conv2D(32,(8,8), strides=4,use_bias =True,bias_initializer='zeros',kernel_initializer = init,activation = 'relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(64,(4,4), strides = 2, use_bias = True, bias_initializer = 'zeros',kernel_initializer = init, activation='relu'))
        model.add(Conv2D(64,(3,3),use_bias= True, bias_initializer = 'zeros', kernel_initializer = init, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform' ))
        model.add(Dense(24, activation = 'relu',kernel_initializer='he_uniform' ))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer = 'he_uniform'))
        model.compile(optimizer=keras.optimizers.RMSprop(lr,rho=0.95), loss = 'mse')
        return model

    def model_save(self):
        self.model.save_weights("model_breakout_dqn.h5")

    def env_re(self):
        return self.env.reset()

    def step(self,a):
        #self.env.render()
        return self.env.step(a)

    def update_target_model(self):
        self.model_t.set_weights(self.model.get_weights())
    def model_load(self):
        self.model.load_weights("model_breakout_dqn.h5")
        self.model_t.load_weights("model_breakout_dqn.h5")


record = []
env_name = 'BreakoutNoFrameskip-v0'
batch_size = 32
count = 0
brain = dqnagent()
learning_start = 10000
st = time.time()
model_saved = False
if model_saved == True:
    brain.model_load()
update_ = 0

if train == True:
    for i in range(100000):
        s = brain.env_re()
        s =cv2.resize(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY),(84,84))
        s = np.reshape(s,(1,84,84))
        s = [s for _ in range(4)]
        s = np.stack(s,axis=3)
        s = np.array(s)

        d = False
        R = 0
        while not d:

            update_ += 1

            a = brain.choose_action(s)
            #print(a)
            s2, r, d, _ = brain.step(a)
            s2 =cv2.resize(cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY),(84,84))
            s2= np.reshape(s2, (1,84,84,1) )
            s2=np.concatenate((s2,s[:,:,:,0:3]),axis=3)
            R +=r
            if d == True :
                r = -1
            brain.add_memory(s,a,r,d,s2)
            s = s2
            count += 1
            if count > learning_start and count %4 == 0:
                brain.learn()
            if d == True:
                record.append(R)
                print(i, R)
                break
            if update_ == 4000:
                update_ = 0
                brain.update_target_model()
        if (i+1) % 100 == 0:
            brain.model_save()
else:
    brain.model.load_weights("model_cart.h5")
record = np.array(record)
plt.plot(record)
plt.xlabel('no of episode')
plt.ylabel('score')
plt.show()
