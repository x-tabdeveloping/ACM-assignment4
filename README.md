# ACM-assignment4
Naive Bayes agent for category learning.

## Setup

Install requirements:

```bash
pip install -r requirements.txt
```

## Simulations

To run simulated studies and parameter recovery, run:

```bash
python3 simulation_study.py
```

To plot results run:

```bash
python3 plot_simulations.py
```

## Empirical data

The Alien data has to be under the following path:

```
 - dat/
    - TrainingAlienData.csv
```

You can then fit models to the data by running:

```bash
python3 fit_to_empirical.py
```

To plot results run:

```bash
python3 plot_empirical.py
```
