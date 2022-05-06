

from pyexpat import model
import pandas as pd
from pprint import pprint
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np
import stan
import sys

# Data Preparation
df = pd.read_csv("https://drive.google.com/uc?export=download&id=1ZH6btsEXM3ekZ3y-m6u_ZxgkF88_zOqW",
                 dtype = {
                     'Lesson' : 'category',
                     'Student': 'category',
                     'KC'     : 'category',
                     'item'   : 'category'
                 })

df['Lesson_number'] = df.Lesson.cat.codes + 1
df['Student_number'] = df.Student.cat.codes + 1
df['KC_number'] = df.KC.cat.codes + 1
df['item_number'] = df.Lesson.cat.codes + 1

items_data = df.loc[(df['firstattempt'] == 1) & (df['Lesson'] == "Splot")]

# New categorical values
items_data['Lesson_number'] = items_data['Lesson_number'].astype("category").cat.codes + 1
items_data['Student_number'] = items_data['Lesson_number'].astype("category").cat.codes + 1
items_data['item_number'] = items_data['Lesson_number'].astype("category").cat.codes + 1

model_data = {
    'K'       : items_data['item'].nunique(),
    'N'       : len(items_data.index),
    'J'       : items_data['Student'].nunique(),
    'y'   : items_data['right'].to_numpy(),
    'kk'    : items_data['item_number'].to_numpy(),
    'jj' : items_data['Student_number'].to_numpy()
}

model_specs = """
data {
  int<lower=1> J;              // number of students
  int<lower=1> K;              // number of questions
  int<lower=1> N;              // number of observations
  int<lower=1,upper=J> jj[N];  // student for observation n
  int<lower=1,upper=K> kk[N];  // question for observation n
  int<lower=0,upper=1> y[N];   // correctness for observation n
}
parameters {
  real mu_beta;                // mean question difficulty
  vector[J] alpha;             // ability for j - mean
  vector[K] beta;              // difficulty for k
  vector<lower=0>[K] gamma;    // discrimination of k
  real<lower=0> sigma_beta;    // scale of difficulties
  real<lower=0> sigma_gamma;   // scale of log discrimination
}
model {
  alpha ~ std_normal();
  beta ~ normal(0, sigma_beta);
  gamma ~ lognormal(0, sigma_gamma);
  mu_beta ~ cauchy(0, 5);
  sigma_beta ~ cauchy(0, 5);
  sigma_gamma ~ cauchy(0, 5);
  y ~ bernoulli_logit(gamma[kk] .* (alpha[jj] - (beta[kk] + mu_beta)));
}
"""

sm = stan.build(program_code=model_specs, data=model_data, random_seed=4051)

fit = sm.sample(num_chains=1, num_samples=1)
irt_res = fit.to_frame()  # pandas `DataFrame, requires pandas

print("End")