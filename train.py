# -*- coding: utf-8 -*-
import gym
import torch
from torch import nn
from torch.autograd import Variable

from model import ActorCritic
from utils import action_to_one_hot, extend_input, state_to_tensor


# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr


def train(rank, args, T, shared_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  action_size = env.action_space.n
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  t = 1  # Thread step counter
  done = True  # Start new episode

  while T.value() <= args.T_max:
    # Sync with shared model at least every t_max steps
    model.load_state_dict(shared_model.state_dict())
    # Get starting timestep
    t_start = t

    # Reset or pass on hidden state
    if done:
      hx = Variable(torch.zeros(1, args.hidden_size))
      cx = Variable(torch.zeros(1, args.hidden_size))
      # Reset environment and done flag
      state = state_to_tensor(env.reset())
      action, reward, done, episode_length = 0, 0, False, 0
    else:
      # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
      hx = hx.detach()
      cx = cx.detach()

    # Lists of outputs for training
    values, log_probs, rewards, entropies = [], [], [], []

    while not done and t - t_start < args.t_max:
      input = extend_input(state, action_to_one_hot(action, action_size), reward, episode_length)
      # Calculate policy and value
      policy, value, (hx, cx) = model(Variable(input), (hx, cx))
      log_policy = policy.log()
      entropy = -(log_policy * policy).sum(1)

      # Sample action
      action = policy.multinomial()
      log_prob = log_policy.gather(1, action.detach())  # Graph broken as loss for stochastic action calculated manually
      action = action.data[0, 0]

      # Step
      state, reward, done, _ = env.step(action)
      state = state_to_tensor(state)
      reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
      done = done or episode_length >= args.max_episode_length
      episode_length += 1  # Increase episode counter

      # Save outputs for training
      [arr.append(el) for arr, el in zip((values, log_probs, rewards, entropies), (value, log_prob, reward, entropy))]

      # Increment counters
      t += 1
      T.increment()

    # Return R = 0 for terminal s or V(s_i; θ) for non-terminal s
    if done:
      R = Variable(torch.zeros(1, 1))
    else:
      _, R, _ = model(Variable(input), (hx, cx))
      R = R.detach()
    values.append(R)

    # Train the network
    policy_loss = 0
    value_loss = 0
    A_GAE = torch.zeros(1, 1)  # Generalised advantage estimator Ψ
    # Calculate n-step returns in forward view, stepping backwards from the last state
    trajectory_length = len(rewards)
    for i in reversed(range(trajectory_length)):
      # R ← r_i + γR
      R = rewards[i] + args.discount * R
      # Advantage A = R - V(s_i; θ)
      A = R - values[i]
      # dθ ← dθ - ∂A^2/∂θ
      value_loss += 0.5 * A ** 2  # Least squares error

      # TD residual δ = r + γV(s_i+1; θ) - V(s_i; θ)
      td_error = rewards[i] + args.discount * values[i + 1].data - values[i].data
      # Generalised advantage estimator Ψ (roughly of form ∑(γλ)^t∙δ)
      A_GAE = A_GAE * args.discount * args.trace_decay + td_error
      # dθ ← dθ + ∇θ∙log(π(a_i|s_i; θ))∙Ψ
      policy_loss -= log_probs[i] * Variable(A_GAE)  # Policy gradient loss
      # dθ ← dθ + β∙∇θH(π(s_i; θ))
      policy_loss -= args.entropy_weight * entropies[i]  # Entropy maximisation loss

    # Optionally normalise loss by number of time steps
    if not args.no_time_normalisation:
      policy_loss /= trajectory_length
      value_loss /= trajectory_length

    # Zero shared and local grads
    optimiser.zero_grad()
    # Note that losses were defined as negatives of normal update rules for gradient descent
    (policy_loss + value_loss).backward()
    # Gradient L2 normalisation
    nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm, 2)

    # Transfer gradients to shared model and update
    _transfer_grads_to_shared_model(model, shared_model)
    optimiser.step()
    if not args.no_lr_decay:
      # Linearly decay learning rate
      _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

  env.close()
