#!/usr/bin/env python
# coding: utf-8

# # Deep Reinforcement Learning: The Cross-Entropy Method
# 
# ## Introduction
# __Reinforcement Learning (RL)__ sits somewhere between __Supervised__ and __Unsupervised Learning.__
# 
# * __Supervised Learning__: Training data $(x_i,y_i),$ $i=1,\dots,$ with the $x_i$ representing data points and $y_i$ the corresponding label.<br>
# Examples: Support Vector Machines (SVMs), Bayes Classifiers, Decision Trees, Neural Networks, etc.
# 
# * __Unsupervised Learning__: One just has data points $x_i,$ $i=1,\dots,n,$ but not label.<br> 
# Examples: Clustering algorithms, Principal Component Analysis (PCA), 
# 
# __Reinforcement Learning__ follows a different route. RL differs from supervised learning in a way that in supervised learning the training data has the answer key with it so the model is trained with the correct answer itself whereas in RL, there is no answer but the reinforcement agent decides what to do to perform the given task. In the absence of a training dataset, it is bound to learn from its experience. <br>
# Examples: 
# * [Game Playing](https://www.deepmind.com/research/highlighted-research/alphago), 
# * [Robotics](https://www.aiperspectives.com/artificial-intelligence-and-robotics), 
# * [Medical Imaging](https://arxiv.org/abs/2103.05115), 
# * [Mathematics (finding constructions/counterexamples to open conjectures)](https://arxiv.org/abs/2104.14516).
# 
# Reinforcement learning can be viewed as being a technique for solving Markov decision processes.
# 
# A __Markov decision process__ is a tuple $(\Omega, A, T, R)$ in which
# * $\Omega$ is the set of __states__,
# * $A$ is a finite set of __actions,__
# * $T$ is a __transition function__ $T:\Omega\times A \times \Omega\to [0,1],$ which, for $T(\omega,a,\omega'),$ gives the probability that action $a$ performed in state $\omega$ will lead to state $\omega',$
# * and $R$ is a __reward function__ defined as $R:\Omega \times A \to \mathbb{R}.$
# 
# A __policy__ $\rho$ is a function $\rho:\Omega \to A,$ which tells the agent which action to perform in any given state. 
# 
# The main goal of learning in this context is to learn (i.e., find) a policy that gathers rewards:
# 
# * DISCOUNTED REWARD: $ \max\mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t]$ with $0\leq \gamma \leq1$ a fixed constant. 
# 
# A lower $\gamma$ makes 
# rewards from the uncertain far future less important for our agent 
# than the ones in the near future that it can be fairly confident 
# about. It also encourages agents to collect reward closer in time 
# than equivalent rewards that are temporally far away in the future.
# 
# The main idea behind Q-learning is that if we had a function
# $Q^*: \Omega \times A \rightarrow \mathbb{R}$, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
# 
# \begin{align}\rho^*(s) = \arg\!\max_a \ Q^*(s, a)\end{align}
# 
# However, we don't know everything about the world, so we don't have
# access to $Q^*$. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# $Q^*$.
# 
# 
# ### Deep Reinforcement Learning
# 
# This brings us to Deep Reinforcement Learning, i.e., Reinforcement Learning using deep neural networks. We discuss one particular method, the so-called __Cross-Entropy__ method. This method is, of course, not always the best method, but it has its own strenghts. The most important ones are the following:
# 
# * __Simplicity__: The cross-entropy method is really simple (<100 lines of code) and intuitive. 
# * __Good convergence__: In environments that don't require complex, multistep policies to be learned and discovered and have short episodes with frequent rewards, cross-entropy usually works very well. 
# 
# We will introduce and discuss this method by discussing the rather popular RL environment FrozenLake. The FrozenLake environment is part of the OpenAI's Gym package, which gives several tools to simulate and compare RL approaches. 
# 
# ### FrozenLake
# 
# You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake (G). The water is mostly frozen (F), but there are a few holes (H) where the ice has melted.
# 
# ![title](Frozen-Lake.png)
# 
# If you step into one of those holes, you’ll fall into the freezing water. At this time, there’s an international frisbee shortage, so it’s absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won’t always move in the direction you intend.
# 
# The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym.spaces

import time


# In[ ]:


env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4", render_mode="ansi")
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)


# Let's start the environment.

# In[ ]:


env.reset()
print(env.render())


# And perform a few steps. The agent has 4 potential actions: 0 (LEFT), 1 (DOWN), 2 (RIGHT), 3 (UP).

# In[ ]:


next_state, reward, episode_is_done, _ , _= env.step(2)
print(env.render())
print(f'Reward = {reward}')


# In principle, we can work out here the optimal policy. (Idea: Moving backwards. Keyword: Bellman-Ford equation, Dynamic Programming). We want to follow here, however, a deep learning aproach.
# 

# The following, so-called [wrapper](https://alexandervandekleut.github.io/gym-wrappers/), for OpenAI's gym allows us to add functionality to environments, such as modifying observations and rewards to be fed to our agent. Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.<br><br>
# We modify the observation space slightly to encode the states as indicator vector ($v\in[0,1]^{16}$), previously it was by integers in $\{0,\dots,15\}.$

# In[ ]:


class MyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
       super(MyWrapper, self).__init__(env)
       self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,), dtype=np.float32) 
          #Our observations/states will be a vector of 16 reals in [0,1]

    def observation(self, observation): #the state vector contains a 1 at index describing the former state
        r = np.copy(self.observation_space.low)
        r[observation] = 1.0
        return r

env = MyWrapper(env)


# In[ ]:


env.reset()
print(env.render())
next_state, reward, episode_is_done, _ , _= env.step(2)
print(next_state)


# Now, we define a neural network.

# In[ ]:


obs_size = env.observation_space.shape[0] # 16
n_actions = env.action_space.n  # 4
HIDDEN_SIZE = 128 #128

net= nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
            #nn.Sigmoid(), 
            nn.Linear(HIDDEN_SIZE, n_actions)
        )

objective = nn.CrossEntropyLoss() #quite standard for classification tasks
optimizer = optim.Adam(params=net.parameters(), lr=0.001)


#  We take the 16 inputs, feed them to a hidden layer with HIDDEN_SIZE nodes, ReLU activation function; there will be four outputs (the actions). The learning rate is set to lr=0.001. As loss function we use the CrossEntropyLoss function, which is defined:
# $$ \ell(x,c) = -x_c + \ln\left(\sum_j \exp(x_j)\right).$$

# Example: $x=(3.2, 1.3,0.2, 0.8)^T.$ $c=0.$ 

# In[ ]:


criterion = nn.CrossEntropyLoss()
output = torch.tensor([[3.2, 1.3,0.2, 0.8]],dtype=torch.float)
target = torch.tensor([0], dtype=torch.long)
print(criterion(output, target))
#print(-output[0][0]+np.log(np.exp(output[0][0])+np.exp(output[0][1])+np.exp(output[0][2])+np.exp(output[0][3])))


# $\ell(x,c)=0$ would be perfect.<br> ![title](CrossEntropyBinaryLoss.png)

# We now want to change our environment so that the agent is performing the action predicted by the NN net.

# In[ ]:


sm = nn.Softmax(dim=1) #Softmax converts the 4-dimensional output vector to a probability distribution

def select_action(state):
        state_t = torch.FloatTensor(np.array([state]))
        act_probs_t = sm(net(state_t))
        act_probs = act_probs_t.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs) #chooses randomly one of the 4 actions according to the probabilities returned by the net
        return action


# In[ ]:


#Here some explantion for the code above:
a=[0.1,0.3,0.4,0,0,0,0,0,0,0,0,0,0,0,0,0]
st=torch.FloatTensor([a])
print(st)
st2=sm(net(st))
print(st2)
print(st2.data.numpy()[0])


# In[ ]:


BATCH_SIZE = 2 #100

GAMMA = 0.9

PERCENTILE = 30 #30
REWARD_GOAL = 0.8

from collections import namedtuple  #more readable tuples

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# In[ ]:


start_time = time.time()

iter_no = 0
reward_mean = 0
full_batch = []
batch = []
episode_steps = []
episode_reward = 0.0
state,_ = env.reset() 
    
while reward_mean < REWARD_GOAL:
        action = select_action(state)
        next_state, reward, episode_is_done, _ , _= env.step(action)

        episode_steps.append(EpisodeStep(observation=state, action=action))
        episode_reward += reward
        
        #print(episode_steps)
        
        if episode_is_done: # Episode finished      
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            
            #print("Episode finished")
            #print(batch)
            #input("Press Enter to continue...") 

            #print(len(batch))
            
            next_state,_ = env.reset()
            episode_steps = []
            episode_reward = 0.0
             
            if len(batch) == BATCH_SIZE: # New set of batches ready --> select "elite"
                reward_mean = float(np.mean(list(map(lambda s: s.reward, batch)))) #compute mean reward (lambda is inline function)
                elite_candidates= batch
                #elite_candidates= batch + full_batch
                returnG = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), elite_candidates))
                reward_bound = np.percentile(returnG, PERCENTILE) #lowest score that is greater than PERCENTILE% of scores in the data set
                                                                  #Keep the highest 100-PERCENTILE %
                #print("Batch finished", returnG, reward_bound)
                
                train_obs = []
                train_act = []
                elite_batch = []
                
                for example, discounted_reward in zip(elite_candidates, returnG):
                        if discounted_reward > reward_bound:
                        #if discounted_reward >= reward_bound:    
                              train_obs.extend(map(lambda step: step.observation, example.steps))
                              train_act.extend(map(lambda step: step.action, example.steps))
                              elite_batch.append(example)
                full_batch=elite_batch
                state=train_obs
                acts=train_act
                
                #Do the training
                if len(full_batch) != 0 : # just in case empty during an iteration
                  state_t = torch.FloatTensor(np.array(state)) #batch of states: [[1.0,0,0,0,0,0,0,0,0,0],[1,...]]               
                  acts_t = torch.LongTensor(acts) # batch of actions: [0,2,3,1,..]               
                  
                  #print(state_t)
                  #print(acts_t)
                  #input("Press Enter to continue...")  
                    
                  optimizer.zero_grad()  #it is good practice to do this, initializing the gradient computations
                  action_scores_t = net(state_t)
                  
                  #print(action_scores_t)
                  #input("Press Enter to continue...") 
                
                  loss_t = objective(action_scores_t, acts_t)
                  loss_t.backward() #computes the gradients
                  optimizer.step() #updates the weights according to the gradients
                  print("%2d: loss=%.3f, reward_mean=%.3f" % (iter_no, loss_t.item(), reward_mean))
                  iter_no += 1
                batch = [] #empty the batch
        state = next_state
        
print("  ----- %.2f seconds ---" % (time.time() - start_time))


# Let's test (explicitely) what our NN learned.

# In[ ]:


env.reset()
state=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
done=0
counter=0
MAXCOUNTER=5

print("\nStep: "+str(counter), "-------")
print(np.size(env.render()), env.render()[0:3])

while done !=1 and counter<MAXCOUNTER:
  state_t = torch.FloatTensor([state])
  action_scores_t = net(state_t)
  act_probs_t = sm(action_scores_t)
  #print(act_probs_t)
  act_probs = act_probs_t.data.numpy()[0]
  proposed_action=np.argmax(act_probs)
  next_state, reward, done, _ , _= env.step(proposed_action)
  state=next_state.tolist()
  counter+=1
  print("\tStep: "+str(counter))
  print(env.render())


# ## Homework
# 
# __Task__: Make adaptations to the above code (i.e. use the cross-entropy method from deep inforcement learning) to find the so-called Longest Increasing Subsequence (LIS) of the sequence [17, 15, 3, 22, 9, 13, 33, 21, 50, 40, 50, 42, 65, 60, 5, 70]. <br>
# 
# The Longest Increasing Subsequence Problem asks to find a subsequence of a given sequence in which the subsequence's elements are sorted in an ascending order and in which the subsequence is as long as possible. For instance, the LIS of the sequence [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15] is [0, 2, 6, 9, 11, 15].
# 
# __Hints__: The following are just hints. You don't need to follow them as long as you find a rather long increasing subsequence with your code. Aim for finding the longest increasing subsequence.<br>
# 
# * You have to think about how you want to represent the states and actions. (For instance, you can represent the current selection of the subsequence by a binary 0/1 vector, where a 1 in position i represents that you have currently chosen the ith term in the sequence; an action could be a binary vector with a single 1 representing the term which you want to add to the subsequence.) 
# 
# * You probably need to program three mini-functions: One that gives you an initial state, one that selects an action as recommened by your NN, and one that performs a specified action given a state.
# 
# * Don't use this line "while reward_mean < REWARD_GOAL:". Substitute is an appropriate way so that you find the longest (or at least a rather long) increasing subsequence of the given input sequence.
# 
# * An episode could create an increasing subsequence. Then the corresponding reward could be the length of this increasing subsequence. 

# 

# This Jupyter Notebook is inspired by https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained and https://gsverhoeven.github.io/post/frozenlake-qlearning-convergence/
