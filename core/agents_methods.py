import torch
import numpy as np
from tqdm import tqdm
import imageio

class Agents_Methods :
    def __init__(self):
        pass

    def train(self, env, episodes):
        for episode in tqdm(range(episodes), desc="Training", ncols=100, ascii=True):
            state, _ = env.reset()
            done = False
               
            step_count = 0
            while not done:  
                action, log_prob, value = self.getaction_ppo(state)
                next_state, reward, done = env.step(action)

                self.store_transition_ppo(
                    state,
                    action,
                    float(reward),
                    float(done),
                    log_prob.detach().squeeze(),
                    value.detach().squeeze())

                self.state = next_state
                    
                state = next_state
                    
                # Update if buffer is full
                if len(self.memory) >= self.buffer_size:
                    self.learn_ppo(state) # We pass the last state for bootstrap
                        
                # No need to reset env here, just need coherence in state transition
                
    def test_agent(self, env, testepisodes):
        total_rewards = []  # List to record the total rewards obtained by the agent
        if self.algo == "dqn":
            self.epsilon = 0  # Exploitation only during testing

        for episode in range(testepisodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                s = self.preprocess_state(state)
                probs = self.nna(s).squeeze(0).detach().cpu().numpy()
                action = int(np.argmax(probs))
                
                state, reward, done = env.step(action)
                total_reward += reward

            # Record the total reward for this episode
            total_rewards.append(total_reward)
            print(f"Episode {episode+1}/{testepisodes} - Total reward: {total_reward}")

        # Average reward over all test episodes
        avg_reward = sum(total_rewards) / testepisodes
        print(f"\nAverage reward over {testepisodes} test episodes: {avg_reward}")
        
    def preprocess_state(self, state):
        # Gymnasium: (obs, info)
        if isinstance(state, tuple):
            state = state[0]

        # If already a tensor, return as-is
        if isinstance(state, torch.Tensor):
            print("State is already a tensor.")
            return state
            
        state = np.array(state, dtype=np.float32)

        if state.ndim == 1:
            # CartPole: (4,) -> (1, 4)
            return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if state.ndim == 2:
            # Batch of 1D states: (batch, features)
            return torch.tensor(state, dtype=torch.float32, device=self.device)

        if state.ndim == 3:
            # Single image: (H, W, C) -> (1, C, H, W) for PyTorch Conv2d
            state = np.transpose(state, (2, 0, 1))  # (C, H, W)
            return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if state.ndim == 4:
            # Batch of images: (batch, H, W, C) -> (batch, C, H, W)
            state = np.transpose(state, (0, 3, 1, 2))
            return torch.tensor(state, dtype=torch.float32, device=self.device)

        raise ValueError(f"Unexpected state shape: {state.shape}")

    def graphic_drone_episode(self, env, agent, filename="drone_test.gif", max_steps=500):
        """
        Visualise un épisode de test du drone et enregistre un GIF.
        
        - env : instance de DroneEnv(render_mode="rgb_array")
        - agent : doit avoir une méthode getaction_ppo(state) -> (action, log_prob, value)
        - filename : chemin du GIF de sortie
        - max_steps : nombre max de steps dans l'épisode
        """
        # Reset de l'environnement
        state, _ = env.reset()
        done = False
        frames = []

        # Barre de progression
        with tqdm(total=max_steps, desc="Création GIF drone", ncols=100, ascii=True) as pbar:
            step = 0
            while not done and step < max_steps:
                # Rendu de l'image actuelle
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                # Sélection de l'action par l'agent (ici PPO discret/continu selon ton implémentation)
                with torch.no_grad():
                    action, _, _ = agent.getaction_ppo(state)

                # Step environnement
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state

                step += 1
                pbar.update(1)

        env.close()

        # Sauvegarde du GIF
        if len(frames) > 0:
            imageio.mimsave(filename, frames, fps=30)
            print(f"GIF sauvegardé sous : {filename}")
        else:
            print("Aucune frame capturée : vérifie que l'env est en render_mode='rgb_array'.")
