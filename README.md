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
a Pendulum and a Cart-Pole system.

For the Pendulum system, run the following script:
```bash
./Pendulum_Identification.py 
```
and for the Cart-Pole system, run:
```bash
./CartPole_Identification.py
```

Optional arguments can be passed when running the previous scripts. The main options
are summarized in the following Table.  

| Command  | Description                                             |
|----------|---------------------------------------------------------|
| `nx`     | Number of states of the NodeREN model                   |
| `nq`     | Number of nonlinearities of the NodeREN model           |
| `t_end`  | End time for the training simulation window: [0, t_end] |
| `method` | Integration method tu use for simulating the NodeREN    |


## Binary classification with NodeRENs

In order to test NodeRENs in benchmark  binary classification problems, run the following script:
```bash
./Binary_Classification.py --dataset [DATASET]
```
where available values for `DATASET` are `double_moons`, `double_circles`, `double_moons`, 
`checker_board`, and `letters`. 


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
