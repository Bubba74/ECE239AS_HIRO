
## ECE239AS Spring 2021 Final Project

### Empirical Paper drawing from [HIRO](https://arxiv.org/abs/1805.08296)

### Files:
File | Description
---- | -----------
basicgym.py | The majority of the Python code.
ECE239AS_Envs | Python module that includes the modified variant of the PyBullet cart-pole environment
figures.py  | A helper Python script for generating figures for the report.
scan.ps1    | Powershell script for evaluating several different algorithms and architectures.

### Installation
Honestly, this probably isn't going to happen. But if so one must install:
- tensorflow
- OpenAI Gym
- PyBullet + envs

### Use-case:
```
python basicgym.py --<ALG> --ActorNN=<ANN> --CriticNN=<CNN>
```
Variable | Value
-------- | -----
ALG      | Can be 'TD3' or 'HIRO'. Remove option for DDPG.
ANN      | Number of neurons in actor network hidden layers.
CNN      | Number of neurons in critic network hidden layers.
