# DEA FDTD Simulation

- Simulate Dielectric Elastomer Actuator (DEA) using Finite-Difference Time Domain (FDTD) method.
- Train the DEA comensator using PyTorch or DiffTaichi.
- [!] Simulation implementation using DiffTaichi is incomplete [!]

### Train

Train the compensator. Training procedures can be found under `result` directory. Various training settings can be given by modifying parameters in `constants.py`.

```bash
~$ python3 torch_models.py --model visco --train t --n_iter 10
```
### Simulate

Simulate the DEA's resulting strain for given voltage. You can also simulate the compensated output of DEA, by using the preprocessed signal.

```bash
# Simulate Maxwell stress compensator
~$ python3 torch_models.py --model maxwell --test t
~$ python3 torch_models.py --model visco --test t --checkpoint $PATH_TO_PRETRAINED_MODEL
~$ python3 test_model.py --model maxwell --test t    # test DiffTaichi implementation
```
