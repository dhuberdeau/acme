# constants:
environment_library = 'gym'

#pre-condition imports
import os

#directory maintenance:
os.chdir('/home/linux4/Documents/project/acme')

# project imports:
import gym
import pyvirtualdisplay
import imageio
import base64
import numpy as np
import sonnet as snt
from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers

#@title load an environment:
environment = gym.make('MountainCarContinuous-v0')
# environment = gym.make('Pendulum-v0')
environment = wrappers.GymWrapper(environment)
environment = wrappers.SinglePrecisionWrapper(environment)
environment_spec = specs.make_environment_spec(environment)

#@title Build agent networks
# Get total number of action dimensions from action spec.
num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

# Create the shared observation network; here simply a state-less operation.
observation_network = tf2_utils.batch_concat

# Create the deterministic policy network.
policy_network = snt.Sequential([
    networks.LayerNormMLP((256, 256, 256), activate_final=True),
    networks.NearZeroInitializedLinear(num_dimensions),
    networks.TanhToSpec(environment_spec.actions),
])

# Create the distributional critic network.
critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP((512, 512, 256), activate_final=True),
    networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
])

# Create a logger for the agent and environment loop.
agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

# Create the D4PG agent.
agent = d4pg.D4PG(
    environment_spec=environment_spec,
    policy_network=policy_network,
    critic_network=critic_network,
    observation_network=observation_network,
    sigma=1.0,
    logger=agent_logger,
    checkpoint=False
)

# Create an loop connecting this agent to the environment created above.
env_loop = environment_loop.EnvironmentLoop(
    environment, agent, logger=env_loop_logger)

# Run a `num_episodes` training episodes.
# Rerun this cell until the agent has learned the given task.
for i_run in range(1, 100):
    env_loop.run(num_episodes=1)

def render(env):
    return env.environment.render(mode='rgb_array')

def display_video(frames, filename='temp.mp4'):
  """Save and display video."""

  # Write video
  with imageio.get_writer(filename, fps=60) as video:
    for frame in frames:
      video.append_data(frame)

  # Read video and display the video
  video = open(filename, 'rb').read()
  b64_video = base64.b64encode(video)
  video_tag = ('<video  width="320" height="240" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())

timestep = environment.reset()
frames = [render(environment)]

while not timestep.last():
  # Simple environment loop.
  action = agent.select_action(timestep.observation)
  timestep = environment.step(action)
  # Render the scene and add it to the frame stack.
  frames.append(render(environment))

# Save and display a video of the behaviour.
display_video(np.array(frames))
