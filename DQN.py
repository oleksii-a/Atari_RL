import gym
import numpy as np
from collections import deque
import random
import operator
from atari_wrappers import wrap_deepmind
import tensorflow as tf
import os

#
class DQN(object):
	def __init__(self, shape, nOutputs, learningRate):

		self.graph = tf.Graph()
		with self.graph.as_default():

			# placeholders
			self.statePH = tf.placeholder("float", [None, shape[0],shape[1],shape[2]])
			self.actionsPH = tf.placeholder(name='action', shape=[None], dtype=tf.int32)
			self.targetsPH = tf.placeholder(name='reward',shape=[None], dtype=tf.float32)
			
			# convolutional layers
			conv1 = tf.layers.conv2d(self.statePH, 32, (8,8), activation=tf.nn.relu, strides = (4,4), kernel_initializer = tf.contrib.layers.variance_scaling_initializer()) # 32
			conv2 = tf.layers.conv2d(conv1, 64, (4,4), activation=tf.nn.relu, strides = (2,2), kernel_initializer = tf.contrib.layers.variance_scaling_initializer()) # 64
			conv3 = tf.layers.conv2d(conv2, 64, (3,3), activation=tf.nn.relu, strides = (1,1), kernel_initializer = tf.contrib.layers.variance_scaling_initializer()) # 64
			
			# dense layer
			flatten = tf.contrib.layers.flatten(conv3)
			hidden = tf.layers.dense(flatten, 512, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
			self.QVals = tf.layers.dense(hidden, nOutputs, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())

			# define loss function and optimizer
			one_hot_mask = tf.one_hot(self.actionsPH, self.QVals.shape[1], on_value=True, off_value=False, dtype=tf.bool)
			selectedQVals = tf.boolean_mask(self.QVals, one_hot_mask)
			self.loss = tf.reduce_mean(tf.square(self.targetsPH - selectedQVals))
			self.optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self.loss)

			# initialize everything
			init = tf.global_variables_initializer()
			self.saver =  tf.train.Saver()		
			self.sess = tf.Session(graph=self.graph)
			self.sess.run(init)

	def train(self, states, actions,targets):
		_, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.statePH: states, self.actionsPH:actions, self.targetsPH: targets})
		return l

	def predict(self,states):
		return self.sess.run(self.QVals, feed_dict={self.statePH: states})

	def save(self, modelname):
		self.saver.save(self.sess, os.path.join(os.getcwd(), modelname))
		return 0;

	def load(self, modelname):
		self.saver.restore(self.sess, './'+ modelname)
		return 0

# replay memory class
class Memory:
    def __init__(self, memsize):
        self._memory = deque([], maxlen=memsize)
        self._idx = 0
        self._memsize = memsize
        self.size = 0

    def addToMemory(self, item):
        self._memory.append(item)
        self.size = len(self._memory)

    def getSamples(self, num):
        n = len(self._memory)
        batch = random.sample(range(0, n - 1), num)
        samples = operator.itemgetter(*batch)(self._memory)
        states = [x[0] for x in samples]
        actions = [x[1] for x in samples]
        rewards = [x[2] for x in samples]
        statesNext = [x[3] for x in samples]
        terminals = [x[4] for x in samples]
        return states, actions, rewards, statesNext, terminals

# hold-on period before start learning
warmUpFrames = 50000

# learning settings
updateInterval = 10000
epsMax= 1
epsMin = 0.1
decaySteps = 1000000
step = (epsMax - epsMin) / decaySteps
eps = epsMax
alpha = 0.00001
gamma = 0.99

# memory and replay settings
batchSize = 32
memLen = 800000
memory = Memory(memLen)

# moving average statistics
rewardHistory = deque(maxlen = 500)
meanReward = []

# initialize environment
env = gym.make('BreakoutDeterministic-v4')
env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False)

#create DQN
kWidth = 84
kHeight = 84
kStackedNum = 4
dqn = DQN([kHeight, kWidth, kStackedNum],env.action_space.n, alpha)
targetDqn = DQN([kHeight, kWidth, kStackedNum],env.action_space.n, alpha)

# make first step and initialize memory
state = env.reset()

#
totalReward = 0
episode = 0
frames = 0
while True:

	# select action using eps greedy strategy
	if np.random.uniform(0, 1) < eps:
		action = env.action_space.sample()
	else:
		action = np.argmax(dqn.predict([np.concatenate(state,axis=-1)]),axis = 1)[0]

	# act and store the transition into replay memory
	stateNext, reward, done, info = env.step(action)
	memory.addToMemory([state, action, reward, stateNext, done])
	state = stateNext

	if memory.size > warmUpFrames:
		
		# sample batch
		states, actions, rewards, statesNext, terminals = memory.getSamples(batchSize)
		states = np.array([np.concatenate(state,axis=-1) for state in states])
		statesNext = np.array([np.concatenate(state,axis=-1) for state in statesNext])

		# compute targets
		futureRewards = np.max(targetDqn.predict(statesNext), axis = 1)
		futureRewards[np.where(np.array(terminals) == True)] = 0
		targets = np.array(rewards) + gamma * futureRewards
		
		# train
		dqn.train(states,actions,targets)

		#
		eps = max(eps-step,epsMin)
	
	totalReward += reward
	frames += 1

	# we store the model and sync target network at once
	if frames % updateInterval == 0:
		dqn.save("dqn")
		targetDqn.load("dqn")

	# episode has ended
	if done == True:

		# calculate reward statistics and print
		rewardHistory.append(totalReward)
		meanReward.append(np.mean(rewardHistory))
		print("Episode:", episode, "Frames:",frames, "Mean reward:", str(meanReward[-1]))

		#
		totalReward = 0
		episode += 1

		# reset for new episode
		state = env.reset()