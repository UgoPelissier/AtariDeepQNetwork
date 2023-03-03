# Atari Deep Q Netwrok

[Reinforcement Learning] pytorch implementation of Deep Q Network (DQN) on Atari Space-Invaders Game

## Installation

- Clone this repo
```
git clone https://github.com/UgoPelissier/AtariDeepQNetwork.git
```

- Install dependencies
```
conda env create -f environment.yml
conda activate atari_dqn
```

This operation can last up to 20 minutes.

## Running

### Run training
```
bash ./scripts/train.sh
```

To view the reward plot, in another terminal run  ```tensorboard --logdir ./src/logs``` and click http://localhost:6006.

You can stop the program (ctrl+C) whenever you are satisfied with the reward obtained. The result is  ```./model/model.pack```.

### Observe the agent playing
Run
```
bash ./scripts/test.sh
```

You can also skip the training part and run
Run
```
bash ./scripts/trained.sh
```
This will use a model already trained (```./model/trained/model.pack```).