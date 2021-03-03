import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BUFFER_SIZE = 10000
learning_rate = 0.0005
gamma = 0.98
batch_size = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = obs.to(device)
        out = self.forward(obs).to(device)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            s_prime_list.append(s_prime)
            done_mask_list.append(done_mask)
        return torch.tensor(s_list, dtype=torch.float).to(device), \
               torch.tensor(a_list).to(device), \
               torch.tensor(r_list).to(device), \
               torch.tensor(s_prime_list, dtype=torch.float).to(device), \
               torch.tensor(done_mask_list).to(device)

    def size(self):
        return len(self.buffer)

def train(q_net, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q_net(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    q_net = Net().to(device)
    q_target = Net().to(device)
    q_target.load_state_dict(q_net.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    #for n_epi in range(10000):
    for n_epi in range(2000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        state = env.reset()
        done = False

        while not done:
            action = q_net.sample_action(torch.from_numpy(state).float(), epsilon)
            s_prime, reward, done, info = env.step(action)
            #import pdb; pdb.set_trace()
            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward/100.0, s_prime, done_mask))
            state = s_prime

            score += reward
            if done:
                break

        if memory.size()>2000:
            train(q_net, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q_net.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    torch.save(q_net.state_dict(), './qnet.pth')
    env.close()
if __name__ == "__main__":
    main()
