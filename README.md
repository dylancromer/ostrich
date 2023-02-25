# ostrich

Ostrich is a library for performing emulation, also called surrogate modeling, using PCA and Gaussian process interpolation. Ostrich was used in [Cromer et al, 2022](https://ui.adsabs.harvard.edu/abs/2022JCAP...10..034C/abstract) to emulate a galaxy cluster weak-lensing model, and its methodology is described there. It is based on an algorithm used in [Heitmann et al, 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...705..156H/abstract) to emulate the power spectrum from cosmological simulations.

The basic premise that you have a function which is expensive to computer, so you want to create an emulator or surrogate which is trained on samples from this expensive function, and can then quickly recreate and interpolate between those samples accurately, with far less computational cost.

## Installation
For now, I recommend installing using `pip`: run
```
pip install --user -e .
```
inside the base directory. To install dependencies, run
```
pip install --user -r requirements.txt

## Usage
