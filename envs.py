import jax
import jax.numpy
import gymnasium as gym
import sys
import argparse
##import d4rl

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type = str, choices = ["mountain_car", "half_cheetah" ,"ant_maze"])
args = parser.parse_args()

if args.config_name == 'mountain_car':

    env = gym.make("MountainCar-v0", render_mode = "human")

elif args.config_name == 'half_cheetah':
    env = gym.make("HalfCheetah-v4", render_mode = "human")

elif args.config_name == "ant_maze":
    env = gym.make('AntMaze_UMaze-v4')

ovservation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()


#def main(args):

#    env = gym.make(args[0], render_mode = "human")
#    ovservation, info = env.reset()

#    for _ in range(1000):
#        action = env.action_space.sample()
#        observation, reward, terminated, truncated, info = env.step(action)

#        if terminated or truncated:
#            observation, info = env.reset()

#    env.close()

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--config_name", type = str, default = "MountainCar-v0")
#    args = parser.parse_args()
#    #config = get_config(args)
#    #main(config)
#    main(args)