NodeREN
# NodeRENs

PyTorch implementation of NodeRENs
as presented in "Unconstrained Parametrization of Dissipative and Contracting Neural Ordinary Differential Equations".

NodeRENs are at the intersection of Neural Ordinary Differential Equations (Neural ODEs) with 
Recurrent Equilibrium Networks (RENs). 

## Technical report

The [Technical Report](docs/Technical_Report.pdf) can be found in the ```docs``` folder.

## Installation

```bash
git clone https://github.com/DecodEPFL/NodeREN.git

cd NodeREN

python setup.py install
```

## System identification with NodeRENs

In the context of system identification, we use NodeRENs for learning the dynamics of 
a Pendulum.
<!-- a Pendulum and a Cart-Pole system. -->

For the Pendulum system, run the following script:
```bash
python ./Pendulum_Identification.py [--nx NX] [--nq NQ] [--n_steps N_STEPS] [--t_end T_END] [--sigma SIGMA]    [--method METHOD] [--seed SEED] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--alpha ALPHA] [--device DEVICE] [--n_cuda N_CUDA] [--learning_rate LEARNING_RATE] [--n_exp N_EXP] [--verbose VERBOSE] [--rtol RTOL] [--atol ATOL] [--steps_integration STEPS_INTEGRATION] [--experiment EXPERIMENT] [--t_stop_training T_STOP_TRAINING] [--GNODEREN GNODEREN] 
```
<!-- and for the Cart-Pole system, run:
```bash
./CartPole_Identification.py
``` -->
 The main options are summarized in the following Table.  

| Command  | Description                                             |
|----------|---------------------------------------------------------|
| `nx`     | Number of states of the NodeREN model                   |
| `nq`     | Number of nonlinearities of the NodeREN model           |
| `t_end`  | End time for the training simulation window: [0, t_end] |
| `method` | Integration method tu use for simulating the NodeREN    |
| `epochs` |(Max) no. of epochs to be used                          |
| `steps_integration`| Number of integration steps used in fixed-steps integration methods|
| `atol`| Absolute tolerance error for adaptive-step integration methods such as 'dopri5' |
| `rtol`| Relative tolerance error for adaptive-step integration methods such as 'dopri5' |

More details about the remaining arguments can be obtained running the following instruction:
```bash
python ./Pendulum_Identification.py --help
```
<!-- Changing properly `method`, `steps_integration` for fixed-steps methods (or `atol`-`rtol` for variable ones), it is possible to obtain a trade-off in terms of precision and computational power.
[TEST](./figures/Comparison_Integration_Methods/Plot_Different_Integration_Methods.pdf)
<p align="center">
<img src="./figures/Comparison_Integration_Methods/Plot_Different_Integration_Methods.pdf" alt="robot_trajectories_before_training" width="600"/> -->
<!-- <img src="./figures/activation_functions/tanh.png" alt="robot_trajectories_after_training_a_neurSLS_controller" width="400"/> -->
<!-- </p>  -->
## Binary classification with NodeRENs

In order to test NodeRENs in benchmark  binary classification problems, run the following script:
```bash
./Binary_Classification.py --dataset [DATASET]
```
where available values for `DATASET` are `double_moons`, `double_circles`, `double_moons`, 
`checker_board`, and `letters`. 

<!-- 
## Examples: 

### Mountains problem (2 robots)

The following gifs show trajectories of the 2 robots before and after the training of a neurSLS controller, 
where the agents that need to coordinate in order to pass through a narrow passage, 
starting from a random initial position marked with &#9675;, sampled from a Normal distribution centered in 
[&#177;2 , -2] with standard deviation of 0.5.

<p align="center">
<img src="./figures/corridorOL.gif" alt="robot_trajectories_before_training" width="400"/>
<img src="./figures/corridor.gif" alt="robot_trajectories_after_training_a_neurSLS_controller" width="400"/>
</p> 

### Swapping problem (12 robots)

The following gifs show the trajectories of the 12 robots before and after the training of a neurSLS controller, 
where the agents swap their initial fixed positions, while avoiding all collisions.

<p align="center">
<img src="./figures/robotsOL.gif" alt="robot_trajectories_before_training" width="400"/>
<img src="./figures/robots.gif" alt="robot_trajectories_after_training_a_neurSLS_controller" width="400"/>
</p> 

### Early stopping of the training
We verify that neurSLS controllers ensure closed-loop stability by design even during exploration. 
Results are reported in the following gifs, where we train the neurSLS controller 
for 0\%, 25\%, 50\% and 75\% of the total number of iterations.

**Mountains problem**: <br>
Note that, for this example, we fix the initial condition to be [&#177;2 , -2] and 
we train for 150 epochs.
<p align="center">
<img src="./figures/corridor0.gif" alt="mountains_0_training" width="200"/>
<img src="./figures/corridor25.gif" alt="mountains_25_training" width="200"/>
<img src="./figures/corridor50.gif" alt="mountains_50_training" width="200"/>
<img src="./figures/corridor75.gif" alt="mountains_75_training" width="200"/>
</p> 

**Swapping problem**:
<p align="center">
<img src="./figures/robots0.gif" alt="robot_trajectories_0_training" width="200"/>
<img src="./figures/robots25.gif" alt="robot_trajectories_25_training" width="200"/>
<img src="./figures/robots50.gif" alt="robot_trajectories_50_training" width="200"/>
<img src="./figures/robots75.gif" alt="robot_trajectories_75_training" width="200"/>
</p>

In both cases, the training is performed for _t_ &in; [0,5].  
Partially trained distributed controllers exhibit suboptimal behavior, but never 
compromise closed-loop stability.

 -->
 

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


## References
[[1]](docs/Technical_Report.pdf) Daniele Martinelli, Clara Galimberti, Ian R. Manchester, 
Luca Furieri, Giancarlo Ferrari-Trecate.
"Unconstrained Parametrization of Dissipative and Contracting Neural Ordinary Differential Equations," 2023.
