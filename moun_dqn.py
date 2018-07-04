import gym
import numpy as np
import keras
import random
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#from gym.wrappers import Monitor
from gym import wrappers
max_ep=20000
train = True


class dqnagent():
    def __init__(self, lr = 0.001, ob_size = 2, action_size = 3, env = 'MountainCar-v0'):
        self.state_size = ob_size
        self.action_size = action_size
        # build model to extimate q value
        self.model = self._build_model(lr)
        # build target model

        self.model_t = self._build_model(lr)
        self.replay_memory = deque(maxlen = 1000000)  # experience replay_memory to store value
        self.reward_memory = deque(maxlen = 100)
        self.env = gym.make(env)
        #self.env = wrappers.Monitor(env, "/tmp/",force = True)
        #self.env.monitor.start("/tmp",resume=True,video_callable=True)
        self.ep_start = 1
        self.ep_stop = .01
        self.ep = 1
        self.ep_decay = 0.9998
        self.batch_size = 32
        self.gamma = 0.99
        self.t_ = 0
        # make target and main model same first then after end of every episode we will update it
        self.update_target_model()

    def add_memory(self,s,a,r,d,s2):
        # adding experience replay memory
        self.replay_memory.append((s, a, r, d, s2))
        self.ep *= self.ep_decay


    def choose_action(self,s):
        ran = np.random.random()

        # you can use non linear decay rate but we will use linear decay for to get good result ####---
        #self.t_ +=1
        #self.ep = self.ep_stop + (self.ep_start - self.ep_stop)*np.exp(-self.ep_decay*self.t_)

        if self.ep >= ran :
            #self.ep -= self.ep_decay
            return self.env.action_space.sample()
        else:
            a = self.model.predict(s)
            return np.argmax(a[0])

    def learn(self):
        st_ = np.zeros((self.batch_size,2))
        st_2 = np.zeros((self.batch_size,2))
        out = np.zeros((self.batch_size,3))
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
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',kernel_initializer='he_uniform' ))
        model.add(Dense(24, activation = 'relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer='he_uniform'))
        model.compile(optimizer=Adam(lr), loss = 'mse')
        return model

    def model_save(self):
        self.model.save_weights("model_cartpole_dqn.h5")

    def env_re(self):
        return self.env.reset()

    def step(self,a):
        self.t_ +=1
        if self.t_ > 10000:
            self.env.render()
        return self.env.step(a)

    def update_target_model(self):
        self.model_t.set_weights(self.model.get_weights())

record = []
env_name = 'MountainCar-v0'
batch_size = 32
count = 0
#env = gym.make(env_name)
#s = env.reset()
brain = dqnagent()
learning_start = 320
#env = wrappers.Monitor(brain.env,force=True, '/tmp/cartpole-experiment-1')
#env = Monitor(env, directory='/tmp/pp',video_callable=False,force=True, write_upon_reset=True)
update_ = 0

if train == True:
    for i in range(max_ep):
        s = brain.env_re()
        s = np.reshape(s, (1, 2) )
        d = False
        R = 0
        for piko in range(200):

            update_ += 1

            a = brain.choose_action(s)
            s2, r, d, _ = brain.step(a)
            #print(d)
            #if d == True:
                #s2 = np.zeros((1,4))
            #else:

            s2 = np.reshape(s2, (1, 2))
            if d == True and r != -1 :
                r = 20
                print('reched')
            R += r
            brain.add_memory(s,a,r,d,s2)
            s = s2
            count += 1
            if count > learning_start:
                brain.learn()
            if d == True:
                # i am updating my target after every episode you can update it after some no of updation depends upon who problem.

                record.append(R)
                print(i, R)
                break
            if update_ == 500:
                update_ =0
                brain.update_target_model()
        if (i+1) % 50 == 0:
            brain.model_save()
        if np.mean(record[-10:0]) > 190:
            print('training complete!!!!')
            break
else:
    brain.model.load_weights("model_cart.h5")
record = np.array(record)
plt.plot(record)
plt.xlabel('no of episode')
plt.ylabel('score')
plt.show()
