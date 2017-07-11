# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import init


class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax()

    # Pass state into model body
    self.fc1 = nn.Linear(self.state_size, hidden_size)
    # Pass previous action, reward and timestep directly into LSTM
    self.lstm = nn.LSTMCell(hidden_size + self.action_size + 2, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, 1)

    # Xavier weight initialisation
    for name, p in self.named_parameters():
      if 'weight' in name:
        init.xavier_uniform(p)
      elif 'bias' in name:
        init.constant(p, 0)

  def forward(self, x, h):
    state, extra = x.narrow(1, 0, self.state_size), x.narrow(1, self.state_size, self.action_size + 2)
    x = self.relu(self.fc1(state))
    h = self.lstm(torch.cat((x, extra), 1), h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    V = self.fc_critic(x)
    return policy, V, (h[0], h[1])
