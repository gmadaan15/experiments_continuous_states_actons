# experiments_continuous_states_actons
Roughly a python ported work from a quite old C++ work on continuous states and actions([link](http://www.incompleteideas.net/papers/SSR-98.pdf)). The work explores Tiles-based(Cmac), Instance-based and Case-based function approximator for both continuous states and actions RL environments like Pendulum and Double Integrator. We mainly focused on the Uniform case of these function approximators and to further extend the project, created a Tile-based that can handle continuous states but discrete actions. We test it on discrete actions gym env like Mountain-car. 

The results that we obtain from the pythonised version are similar to the old results in the original paper. 
## Requirements before running.
To run this, make sure that you have python3.8.3 versions and you have run the below commands in sequence.

### Making the custom environments as gym environments so that its easy to import as well as run.
```
pip3 install requirements.txt
pip3 install -e ./gym-env/
```

To see the generated results on tensorboard in a web browser, run the below command:
```
tensorboard --logdir=runs

```
