# Converted from RL_Alpers.ipynb adding more printings and selecting 8x8  (grid) 

# # Deep Reinforcement Learning: The Cross-Entropy Method
#  via: jupyter --nbconvert RL_Alpers.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym.spaces

import time


# In[ ]:  IMPORTANT STEP

env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8", render_mode="ansi")
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space, "with postions = %d" % env.observation_space.n)
Total_posi = env.observation_space.n

# Let's start the environment.

# In[ ]:


env.reset()
print(env.render())


# And perform a few steps. The agent has 4 potential actions: 0 (LEFT), 1 (DOWN), 2 (RIGHT), 3 (UP).

# In[ ]:


next_state, reward, episode_is_done, _ , _= env.step(2)
print(env.render())
print(f'Reward = {reward}')
print( next_state, reward, episode_is_done, '<== next_state, reward, episode_is_done')

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
# output = torch.tensor([[3.2, 1.3,0.2, 0.8]],dtype=torch.float)
# target = torch.tensor([0], dtype=torch.long)
# print(criterion(output, target))
# #print(-output[0][0]+np.log(np.exp(output[0][0])+np.exp(output[0][1])+np.exp(output[0][2])+np.exp(output[0][3])))


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
# a=[0.1,0.3,0.4,0,0,0,0,0,0,0,0,0,0,0,0,0]
# st=torch.FloatTensor([a])
# print(st)
# st2=sm(net(st))
# print(st2)
# print(st2.data.numpy()[0])


# In[ ]:


from collections import namedtuple  #more readable tuples

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# In[ ]:


start_time = time.time() #==================== MAIN 


BATCH_SIZE = 200  #100
GAMMA = 0.7
MAXCOUNTER=500

PERCENTILE = 30 #30
REWARD_GOAL = 0.91
iter_no = 0
reward_mean = 0
full_batch = []
batch = []
episode_steps = []
episode_reward = 0.0
state,_ = env.reset(); kt=0; pr=0
    
while reward_mean < REWARD_GOAL:
        
        action = select_action(state)
        next_state, reward, episode_is_done, _ , _= env.step(action)

        episode_steps.append(EpisodeStep(observation=state, action=action))
        episode_reward += reward;            #print(episode_steps)
        
        if episode_is_done: # Episode finished  

            if (next_state[-1]==1) :
                kt += 1 # counter of batch
                if ((kt-1) % 40==0) and (iter_no>=99 and iter_no<=100):
                     pr=1
                     print('\t%2d' % kt, ':', action,np.int16(state), '\n\t>', np.int16(next_state));

              
            batch.append(Episode(reward=episode_reward, steps=episode_steps))            
            #print(batch);   
            next_state,_ = env.reset();         episode_steps = [];            episode_reward = 0.0
             
            if len(batch) == BATCH_SIZE: # New set of batches ready --> select "elite"
                               
                reward_mean = float(np.mean(list(map(lambda s: s.reward, batch)))) #compute mean reward (lambda is inline function)
                elite_candidates= batch
                if pr==1:
                        print("%3d Past episode checking : reward %.2f or avg %.2f " % (iter_no,episode_reward,reward_mean), 
                               "Batch size= ",len(batch), f'({kt:d} times to G)');   
                pr=0; kt=0
                # if reward_mean>.970:
                #         input("Press Enter to continue..."); 
                # #elite_candidates= batch + full_batch
                returnG = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), elite_candidates))
                reward_bound = np.percentile(returnG, PERCENTILE) #lowest score that is greater than PERCENTILE% of scores in the data set
                                                                  #Keep the highest 100-PERCENTILE %
                #print("Batch finished", returnG, reward_bound)
                
                state = [];            acts = [];                elite_batch = []
                
                for example, discounted_reward in zip(elite_candidates, returnG):
                        if discounted_reward > reward_bound:                         #if discounted_reward >= reward_bound:    
                              state.extend(map(lambda step: step.observation, example.steps))
                              acts.extend(map(lambda step: step.action, example.steps))
                              elite_batch.append(example)
                full_batch=elite_batch           
                
                #Do the training
                if len(full_batch) != 0 : # just in case empty during an iteration
                  state_t = torch.FloatTensor(np.array(state)) #batch of states: [[1.0,0,0,0,0,0,0,0,0,0],[1,...]]               
                  acts_t = torch.LongTensor(acts) # batch of actions: [0,2,3,1,..]                                 
                  #print(state_t);                  #print(acts_t);                  #input("Press Enter to continue...")  
                    
                  optimizer.zero_grad()  #it is good practice to do this, initializing the gradient computations
                  action_scores_t = net(state_t)
                  
                  #print(action_scores_t);                  #input("Press Enter to continue...")                 
                  loss_t = objective(action_scores_t, acts_t)
                  loss_t.backward() #computes the gradients
                  optimizer.step() #updates the weights according to the gradients
                  if iter_no % 20==0:
                       print("epo=%3d: loss=%.3f, reward_mean=%.3f" % (iter_no, loss_t.item(), reward_mean),
                             'selecting %d / %d' %(len(full_batch),BATCH_SIZE))
                  iter_no += 1
                batch = [] #empty the batch
        state = next_state
        
print("  %d Traning Steps done ----- %.2f seconds ---" % (iter_no, time.time() - start_time))


# Let's test (explicitely) what our NN learned.

# In[ ]:


env.reset()
# state=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
 
state=np.zeros(Total_posi, dtype=int); state[0]=1
state=state[None,].tolist();   done=0;  counter=0

print("\nStep: "+str(counter), "-------");  print(env.render())

while done !=1 and counter<MAXCOUNTER:
  if (counter+1) % 10==0:
        print('%2d' % counter, end='\r')
  else:
        print('%2d_' % counter, end="");  
  #print(state.dtype) #state #torch.from_numpy(state) #   b
  state_t =  torch.FloatTensor([state]) #state #
  #print(state_t.dtype)
  #print(state_t.dtype, state_t.shape)
  action_scores_t = net(state_t)
  act_probs_t = sm(action_scores_t)
  #print(act_probs_t)
  act_probs = act_probs_t.data.numpy()[0]
  proposed_action=np.argmax(act_probs)
  next_state, reward, done, _ , _= env.step(proposed_action)
  state=next_state.tolist()
  counter+=1
print("\tStep: "+str(counter))
print(env.render(), np.int16(state), '<- Last state')
if (next_state[-1]==1) :
     print('Success :-) *** using %d steps' % counter)
else:
     print(f'MAXCOUNTER = {MAXCOUNTER} reached but need more steps :-(')


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
