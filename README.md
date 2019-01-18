# Deep Q-learning (DQN) for Multi-agent Reinforcement Learning (RL)

DQN implementation for two multi-agent environments: `agents_landmarks` and `predators_prey` (See [details.pdf](https://github.com/mohammadasghari/dqn-multi-agent-rl/blob/master/details.pdf) for a detailed description of these environments).

## Code structure
- `./environments/`: folder where the two environments (`agents_landmarks` and `predators_prey`) are stored. 
    1) `./environments/agents_landmarks`: in this environment, there exist ***n*** agents that must cooperate through actions to reach a set of ***n*** landmarks  in a two dimensional discrete ***k***-by-***k*** grid environment. 
    2) `./environments/predators_prey`: in this environment, ***n*** agents (called predators) must cooperate with each other to capture one prey in a two dimensional discrete ***k***-by-***k*** grid environment.
- `./dqn_agent.py`: contains code for the implementation of DQN and its extensions (Double DQN, Dueling DQN, DQN with Prioritized Experience Replay) (See [details.pdf](https://github.com/mohammadasghari/dqn-multi-agent-rl/blob/master/details.pdf) for a detailed description of the DQN and its extensions).
- `./brain.py`: contains code for the implementation of neural networks required for DQN (See [details.pdf](https://github.com/mohammadasghari/dqn-multi-agent-rl/blob/master/details.pdf) for a detailed description of the neural network implementation).
- `./uniform_experience_replay.py`: contains code for the implementation of Uniform Experience Replay (UER) which can be used in DQN.
- `./prioritized_experience_replay.py`: contains code for the implementation of Prioritized Experience Replay (PER) which can be used in DQN.
- `./sum_tree.py`: contains code for the implementation of sum tree data structure which is used in Prioritized Experience Replay (PER).
- `./agents_landmarks_multiagent.py`: contains code for applying DQN to the `agents_landmarks` environment.
- `./predators_prey_multiagent.py`: contains code for applying DQN to the `predators_prey` environment.
- `./results_agents_landmarks/`: folder where the results (neural net weights, rewards of the episodes, videos, figures, etc.) for the `agents_landmarks` environment are stored. 
- `./results_predators_prey/`: folder where the results (neural net weights, rewards of the episodes, videos, figures, etc.) for the `predators_prey` environment are stored. 
- `./details.pdf`: a pdf file including a detailed description of the DQN and its extensions, the environments, and the neural network implementation.

## Results
#### Predators and Prey Environment
In this environment, the prey is captured when one predator moves to the location of the prey while the other predators occupy, for support, the neighboring cells of the prey's location.
##### Fixed prey (mode 0) 
 <img src="/results_predators_prey/videos/prey_mode_0.gif" height="400px" width="400px" >

##### Random prey (mode 1) 
 <img src="/results_predators_prey/videos/prey_mode_1.gif" height="400px" width="400px" >
 
##### Random escaping prey (mode 2) 
  <img src="/results_predators_prey/videos/prey_mode_2.gif" height="400px" width="400px" >

#### Agents and Landmarks Environment

##### 10 agents and 10 landmarks
<img src="/results_agents_landmarks/videos/10_10.gif" height="400px" width="400px" >

##### 16 agents and 16 landmarks
<img src="/results_agents_landmarks/videos/16_16.gif" height="400px" width="400px" >

### Todos

 - Write required dependencies and installation steps
 - ...
