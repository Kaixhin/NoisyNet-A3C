# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import init, Parameter
from torch.nn import functional as F
from torch.autograd import Variable


# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))

  def reset_parameters(self):
    init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
    init.constant(self.bias, 0.0017)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)


class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size, no_noise):
    super(ActorCritic, self).__init__()
    self.no_noise = no_noise
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax()

    # Pass state into model body
    self.fc1 = nn.Linear(self.state_size, hidden_size)
    # Pass previous action, reward and timestep directly into LSTM
    self.lstm = nn.LSTMCell(hidden_size + self.action_size + 2, hidden_size)
    if no_noise:
      self.fc_actor = nn.Linear(hidden_size, self.action_size)
      self.fc_critic = nn.Linear(hidden_size, 1)
    else:
      self.fc_actor = NoisyLinear(hidden_size, self.action_size)
      self.fc_critic = NoisyLinear(hidden_size, 1)

    # Xavier weight initialisation
    for name, p in self.named_parameters():
      if 'weight' in name:
        init.xavier_uniform(p)
      elif 'bias' in name:
        init.constant(p, 0)
    if not no_noise:
      # Noisy linear initialisation
      self.fc_actor.reset_parameters()
      self.fc_critic.reset_parameters()

  def forward(self, x, h):
    state, extra = x.narrow(1, 0, self.state_size), x.narrow(1, self.state_size, self.action_size + 2)
    x = self.relu(self.fc1(state))
    h = self.lstm(torch.cat((x, extra), 1), h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    V = self.fc_critic(x)
    return policy, V, (h[0], h[1])

  def sample_noise(self):
    if not self.no_noise:
      self.fc_actor.sample_noise()
      self.fc_critic.sample_noise()

  def remove_noise(self):
    if not self.no_noise:
      self.fc_actor.remove_noise()
      self.fc_critic.remove_noise()
