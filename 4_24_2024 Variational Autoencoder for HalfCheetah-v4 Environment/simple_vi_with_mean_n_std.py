# %%

import numpy as np
import jax.numpy as jnp
import gymnasium as gym
import matplotlib.pyplot as plt
import jax
from IPython.display import display, clear_output
from jax import value_and_grad
import jax.random as random
from jax.nn import relu, sigmoid,tanh, softplus

#%%

env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
obs, info = env.reset()
action = env.action_space.sample()
# %%

hidden = 32
weights = {
  "encoder_mean": jnp.asarray(np.random.uniform(-1, 1, size=(*env.observation_space.shape, hidden))),
  "encoder_log_var": jnp.asarray(np.random.uniform(-1, 1, size=(*env.observation_space.shape, hidden))),
  "transition_mean": jnp.asarray(np.random.uniform(-1, 1, size=(hidden, hidden))),
  "transition_log_var": jnp.asarray(np.random.uniform(-1, 1, size=(hidden, hidden))),
  "decoder_mean": jnp.asarray(np.random.uniform(-1, 1, size=(hidden, *env.observation_space.shape))),
  "decoder_log_var": jnp.asarray(np.random.uniform(-1, 1, size=(hidden, *env.observation_space.shape))),
}
learning_rate = 0.0001
key = random.PRNGKey(0)  # Initialize a random key

# %%

def posterior(params, observation):
  mu = observation @ params['encoder_mean']
  log_var = observation @ params['encoder_log_var']
  std = softplus(log_var)  # Ensures variance is positive and more stable
  
  return mu, std

def transition(params, z) -> jax.Array:
  mu = z @ params['transition_mean']
  log_var = z @ params['transition_log_var']
  std = softplus(log_var)  # Ensures variance is positive and more stable
  
  return mu, std

def decode(params, z) -> jax.Array:
  mu = z @ params['decoder_mean']
  log_var = z @ params['decoder_log_var']
  std = softplus(log_var)  # Ensures variance is positive and more stable

  
  return mu, std

def reparameterize(key, mean, std = 1.0):
  #fix std 1
  eps = random.normal(key, shape = mean.shape)
  return mean + std * eps
#NOTE for now just use the mean

# %%
def kl(s1_mean, s1_std, s2_mean, s2_std):
  return jnp.mean(0.5 * (jnp.square(s1_mean - s2_mean) / jnp.square(s2_std) + jnp.square(s1_std) / jnp.square(s2_std) - jnp.log(jnp.square(s1_std) / jnp.square(s2_std)) - 1))
#NOTE kl divergence with mu 1 sigma 1 mu 2 sigma 2 (of the same D dimentional)
#NOTE covariance matrix 
#NOTE Plug in to the formula With sigmas vector being 1, should be similar to MSE

def ce(o1, o2_mean, o2_std):
  n = o1.shape[0]
  square_diff = jnp.square(o1 - o2_mean)
  return jnp.mean(-n/2 * jnp.log(2 * jnp.pi * jnp.square(o2_std)) + jnp.sum(square_diff) / (2 * jnp.square(o2_std)))
#NOTE when i have the both mu and sigma, i should compute gaussian log likelihood, 

def loss_n_predict(params, o_t, o_tp1):
  # Compute states and predictions
  global key
  key, subkey = random.split(key)  # Split the key to maintain statelessness NOTE change the key setting
  (s_t_mean, s_t_std) = posterior(params, o_t)  # Current "real" state's mean
  s_t_z = reparameterize(subkey, s_t_mean, s_t_std)  # current "real" state's z
  (prior_s_tp1_mean, prior_s_tp1_std) = transition(params, s_t_z) # Next assumed state's mean NOTE i should also sample here instead of linear transformation to make it st+1
  prior_s_tp1_z = reparameterize(subkey, prior_s_tp1_mean, prior_s_tp1_std) # Next assumed state's z
  (posterior_s_tp1_mean, posterior_s_tp1_std) = posterior(params, o_tp1)  # Next "real" state's mean
  #posterior_s_tp1_z = reparameterize(subkey, posterior_s_tp1_mean, posterior_s_tp1_std)  # Next "real" state's z
  (o_hat_tp1_mean, o_hat_tp1_std) = decode(params, prior_s_tp1_z)  # Next assumed observation
  print(o_hat_tp1_mean)
  print(o_hat_tp1_std)
  o_hat_tp1 = reparameterize(subkey, o_hat_tp1_mean, o_hat_tp1_std)
  # Compute loss


  kl_val = kl(posterior_s_tp1_mean, posterior_s_tp1_std, prior_s_tp1_mean, prior_s_tp1_std)
  print(kl_val)
  ce_val = ce(o_tp1, o_hat_tp1_mean, o_hat_tp1_std)
  print(ce_val)

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
o_t = obs   #the current observation
o_tp1 = obs #the next observation
loss, o_hat_tp1, gradients = evaluate_and_grad(weights, o_t, o_tp1)
weights = {k: v - learning_rate * gradients[k] for k, v in weights.items()}
print(loss)
#print(gradients)
print(o_hat_tp1)

next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
if terminated or truncated:
      observation, info = env.reset()
else:
    obs = next_obs




# %%
num_steps = 200

vfe_values = []
for step in range(200):
  next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
  # example data
  o_t = obs   #the current observation
  o_tp1 = obs #the next observation
  # dynamics
  loss, o_hat_tp1, gradients = evaluate_and_grad(weights, o_t, o_tp1)
  ## Gradient clipping
  #gradients = {k: jnp.clip(v, -1.0, 1.0) for k, v in gradients.items()}
  weights = {k: v - learning_rate * gradients[k] for k, v in weights.items()}
  #print(weights)
  #print(gradients)
  #time.sleep(5)
  vfe_values.append(loss)
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
# Plot the metrics of losses.
plt.figure(figsize=(12, 4))
plt.plot(vfe_values, label='VFE')
plt.title('Variational Free Energy over Time')
plt.xlabel('Time Step')
plt.ylabel('VFE')
plt.legend()

plt.tight_layout()
plt.show()
# %%





