# %%

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import expit as sigmoid
from jax import grad
import gymnasium as gym
import matplotlib.pyplot as plt
import jax
from IPython.display import display, clear_output
import time
from jax import value_and_grad

#%%

env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
obs, info = env.reset()
action = env.action_space.sample()
# %%

obs.shape
# %%
action.shape
# %%

env.observation_space.shape
env.render().shape
# %%

hidden = 32
weights = {
  "encoder": jnp.asarray(np.random.uniform(-1, 1, size=(*env.observation_space.shape, hidden))),
  "transition": jnp.asarray(np.random.uniform(-1, 1, size=(hidden, hidden))),
  "decoder": jnp.asarray(np.random.uniform(-1, 1, size=(hidden, *env.observation_space.shape))),
}
learning_rate = 0.0001

# %%

weights["encoder"].shape


# %%

def posterior(params, observation):
  return observation @ params['encoder']

def transition(params, state_tm1) -> jax.Array:
  return state_tm1 @ params['transition']

def decode(params, state) -> jax.Array:
  return state @ params['decoder']

def kl(s1, s2):
  epsilon = 1e-8 #smal number to prevent division by 0 ?is this necessary?
  P = s1
  Q = s2 
  #normalize
  #P /= P.sum()
  #Q /= Q.sum()
  logval = jnp.clip(P/Q, epsilon, 1.0)
  
  return jnp.sum(P *jnp.log(logval)) # NOTE: code this up

def ce(o1, o2):
  P = o1
  Q = o2
  #normalize
  P /= P.sum()
  Q /= Q.sum()
  epsilon = 1e-8
  Q = jnp.clip(Q, epsilon, 1.0)  # Clip Q to avoid log(0)
  return -jnp.sum(P * jnp.log(Q)) # NOTE: code this up


def loss_n_predict(params, o_t, o_tp1):
  # Compute states and predictions
  s_t = posterior(params, o_t)  # Current "real" state
  prior_s_tp1 = sigmoid(transition(params, s_t))  # Next assumed state
  posterior_s_tp1 = sigmoid(posterior(params, o_tp1))  # Next "real" state
  o_hat_tp1 = jnp.tanh(decode(params, prior_s_tp1))  # Next assumed observation

  # Compute loss
  kl_val = kl(posterior_s_tp1, prior_s_tp1)
  ce_val = ce(o_tp1, o_hat_tp1)
  VFE = kl_val + ce_val

  return VFE, o_hat_tp1  # Return both loss and predictions

# This function will return the loss, predictions, and gradients
def evaluate_and_grad(params, o_t, o_tp1):
  # This function only calculates the gradient of the first output (loss)
  #value_and_grad(..., has_aux=True) is used here.
  #This tells JAX to treat additional outputs of combined_function (in this case, o_hat_tp1) as auxiliary data
  # that do not contribute to the gradient computation but are still returned from the function.
  loss_and_grad = value_and_grad(loss_n_predict, argnums=0, has_aux=True)
  (loss, predictions), grads = loss_and_grad(params, o_t, o_tp1)
  return loss, predictions, grads


 # %% test loss and gradient
# o_t = obs   #the current observation
# o_tp1 = obs #the next observation
# loss, o_hat_tp1, gradients = evaluate_and_grad(weights, o_t, o_tp1)
# weights = {k: v - learning_rate * gradients[k] for k, v in weights.items()}
# print(gradients)
# print(o_hat_tp1)
# %%
# #Testing visualize the actual and predicted observations
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.bar(range(len(o_tp1)), o_tp1, color='skyblue')
# plt.title('Bar Plot of Actual Observation Vector')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)


# plt.subplot(1, 2, 2)
# plt.bar(range(len(o_hat_tp1)), o_hat_tp1, color='skyblue')
# plt.title('Bar Plot of predicted Observation Vector')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)

# all_values = np.concatenate([o_tp1, o_hat_tp1])
# lower_bound = min(all_values)
# upper_bound = max(all_values)
# abs_max = max(abs(lower_bound), abs(upper_bound))
# plt.subplot(1, 2, 1).set_ylim([-abs_max, abs_max])  # Set symmetrical limits around zero
# plt.subplot(1, 2, 2).set_ylim([-abs_max, abs_max])  # Set symmetrical limits around zero
# plt.show()





# %%

for step in range(200):
  next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
  # example data
  o_t = obs   #the current observation
  o_tp1 = obs #the next observation
  # dynamics
  loss, o_hat_tp1, gradients = evaluate_and_grad(weights, o_t, o_tp1)
  weights = {k: v - learning_rate * gradients[k] for k, v in weights.items()}
  print(weights)
  print(gradients)
  #time.sleep(5)
# Visualize the actual and predicted observations

  clear_output(wait = True)
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  plt.bar(range(len(o_tp1)), o_tp1, color='skyblue')
  plt.title('Bar Plot of Actual Observation Vector')
  plt.xlabel('Index')
  plt.ylabel('Value')
  plt.grid(True)


  plt.subplot(1, 2, 2)
  plt.bar(range(len(o_hat_tp1)), o_hat_tp1, color='skyblue')
  plt.title('Bar Plot of predicted Observation Vector')
  plt.xlabel('Index')
  plt.ylabel('Value')
  plt.grid(True)

# Dynamically adjust y-axis limits
  all_values = np.concatenate([o_tp1, o_hat_tp1])
  lower_bound = min(all_values)
  upper_bound = max(all_values)
  abs_max = max(abs(lower_bound), abs(upper_bound))
  plt.subplot(1, 2, 1).set_ylim([-abs_max, abs_max])  # Set symmetrical limits around zero
  plt.subplot(1, 2, 2).set_ylim([-abs_max, abs_max])  # Set symmetrical limits around zero
  
  plt.suptitle(f'Step {step + 1}')
  display(plt.gcf())
  plt.close()


  if terminated or truncated:
        observation, info = env.reset()
  else:
     obs = next_obs

env.close()
# %%





