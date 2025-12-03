import numpy as np
import torch
from torch.distributions import Normal
from collections import deque
from core.agents_methods import Agents_Methods

class PPOAgent(Agents_Methods):
    def __init__(self, 
            actor_nn,
            critic_nn,
            n_actions,
            buffer_size=512,
            batch_size=64,
            nb_epochs=4,
            gamma=0.99,
            clip_value=0.2,
            lambda_gae=0.95,
            clip_vloss=True,
            entropy_bonus=True,
            shuffle=True,
            action_std=0.5
            target_kl=0.03
        ):
        super().__init__()

        self.device = torch.device("cpu") # Force CPU for compatibility
        self.nna = actor_nn.to(self.device)
        self.nnc = critic_nn.to(self.device)
        self.n_actions = n_actions
        self.action_std = torch.full((n_actions,), action_std, device=self.device) # For continuous action spaces (fixed to start)
        self.loss_fct = torch.nn.MSELoss()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.memory = deque(maxlen=self.buffer_size)
        self.gamma = gamma
        self.clip_value = clip_value
        self.lambda_gae = lambda_gae
        self.c1 = 0.5
        self.c2 = 0.01
        self.ent_bonus = entropy_bonus
        self.shuffle = shuffle
        self.target_kl = target_kl
        
    @torch.no_grad() # We don't want to compute gradients when selecting actions, because we are not training
    def getaction_ppo(self, state):
        """Select action according to current policy and return action, log probability and value."""
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean = self.nna(state)
        dist = Normal(mean, self.action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1) 
        value = self.nnc(state)
        action_np = action.cpu().numpy()[0]
        return action_np, log_prob, value

    def store_transition_ppo(self, state, action, reward, done, log_prob_old, value_old):
        """Store transition in memory."""
        self.memory.append((state, action, reward, done, log_prob_old, value_old))
        
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE) and returns."""
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        values = torch.cat((values, torch.tensor([next_value], dtype=torch.float32, device=self.device)))
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[t])
            advantages[t] = gae
            
        returns = advantages + values[:-1] # R_t = A_t + V(s_t)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalization
        return advantages, returns
        
    def learn_ppo(self, last_state):
        """Update policy and value networks using PPO algorithm."""
        states, actions, rewards, dones, old_log_probs, values = zip(*self.memory) # Learning on a complete rollout

        states = np.array(states, dtype=np.float32)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = np.array(actions, dtype=np.float32)  # [T, n_actions]
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device) # unsqueeze not needed, already 1D for the compute_gae, dones and rewards are not used in the loss directly
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)
        last_state = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            last_value = self.nnc(last_state).squeeze(-1)

        next_value = 0 if dones[-1] else last_value # if last state is done, so last value is 0
        advantages, returns = self.compute_gae(rewards, values, dones, next_value) # Bootstrap value for the last state
        
        size = len(rewards) # To ensure we go through the entire trajectory, not just with buffer_size
        
        for epoch in range(self.nb_epochs):
            # indices shuffle or not
            if self.shuffle:
                idx = torch.randperm(size, device=self.device)
            else:
                idx = torch.arange(size, device=self.device)
                
            for start in range(0, size, self.batch_size):
                end = min(start + self.batch_size, size)
                batch_idx = idx[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # New log probs and values
                mean = self.nna(batch_states)
                dist = Normal(mean, self.action_std)
                log_probs_all = dist.log_prob(batch_actions)
                new_log_probs = log_probs_all.sum(dim=-1)
                new_values = self.nnc(batch_states).squeeze(-1)

                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # Compute approximation KL
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    if approx_kl > self.target_kl:
                        break

                # PPO loss
                surr1 = ratio * batch_advantages.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * batch_advantages.detach()

                # Optional: clipped value loss
                if self.clip_vloss:
                    v_loss_unclipped = (new_values - batch_returns) ** 2
                    v_clipped = batch_values + torch.clamp(new_values - batch_values, -self.clip_value, self.clip_value)
                    v_loss_clipped = (v_clipped - batch_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * (new_values - batch_returns).pow(2).mean()

                # Optional: entropy bonus for exploration
                if self.ent_bonus:
                    entropy = dist.entropy().mean()
                    actor_loss = -torch.min(surr1, surr2).mean() - self.c2 * entropy
                    critic_loss = self.c1 * self.loss_fct(new_values, batch_returns)
                else:
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = self.c1 * self.loss_fct(new_values, batch_returns)

                # Update networks with backpropagation
                self.nna.optimizer.zero_grad()
                actor_loss.backward()
                self.nna.optimizer.step()

                self.nnc.optimizer.zero_grad()
                critic_loss.backward()
                self.nnc.optimizer.step()
        
        # Do not forget to clear memory
        self.memory.clear()