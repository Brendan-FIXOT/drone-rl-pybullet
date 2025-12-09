import time
import torch
import numpy as np
from tqdm import tqdm
from tqdm import trange
import imageio

class Agents_Methods :
    def __init__(self):
        pass

    def train_ppo(self, env, episodes):
        for episode in tqdm(range(episodes), desc="Training", ncols=100, ascii=True):
            state, _ = env.reset()
            done = False
               
            while not done:  
                action, log_prob, value = self.getaction_ppo(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

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
    
    def train_ppo_vectorized(self, envs, total_timesteps, num_steps):
        """
        train PPO with vectorized environments.
        - envs : vectorized environments
        - total_timesteps : total timesteps to train
        - num_steps : number of steps to collect per environment before update
        """
        device = self.device

        num_envs = envs.num_envs
        obs_shape = envs.single_observation_space.shape
        act_shape = envs.single_action_space.shape

        rollout_batch_size = num_steps * num_envs
        minibatch_size = self.batch_size

        assert rollout_batch_size % minibatch_size == 0, \
            f"rollout_batch_size {rollout_batch_size} doit être divisible par minibatch_size {minibatch_size}"

        num_minibatches = rollout_batch_size // minibatch_size
        num_updates = total_timesteps // rollout_batch_size

        print(f"[PPO] num_envs={num_envs} | num_steps={num_steps} | "
            f"rollout_batch_size={rollout_batch_size} | minibatch_size={minibatch_size} | "
            f"num_minibatches={num_minibatches} | num_updates={num_updates}")

        # Buffers
        obs_buf      = torch.zeros((num_steps, num_envs) + obs_shape, dtype=torch.float32, device=device)
        actions_buf  = torch.zeros((num_steps, num_envs) + act_shape, dtype=torch.float32, device=device)
        logprob_buf  = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        rewards_buf  = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        dones_buf    = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        values_buf   = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        # Initial reset
        next_obs, _ = envs.reset(seed=0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

        global_step = 0

        start_time = time.time()
        global_step = 0

        for update in trange(num_updates, desc="PPO Updates", ncols=100, ascii=True):

            # Collect rollout
            for step in range(num_steps):
                global_step += num_envs

                obs_buf[step] = next_obs
                dones_buf[step] = next_done

                actions = []
                logprobs = []
                values = []

                for env_i in range(num_envs):
                    state_i = next_obs[env_i].cpu().numpy()
                    action_i, log_prob_i, value_i = self.getaction_ppo(state_i)
                    actions.append(action_i)
                    logprobs.append(log_prob_i.item())
                    values.append(value_i.item())

                actions = np.array(actions, dtype=np.float32)
                actions_t = torch.tensor(actions, dtype=torch.float32, device=device)

                actions_buf[step] = actions_t
                logprob_buf[step] = torch.tensor(logprobs, dtype=torch.float32, device=device)
                values_buf[step] = torch.tensor(values, dtype=torch.float32, device=device)

                next_obs_np, rewards_np, terms, truncs, infos = envs.step(actions)
                next_done = torch.tensor(np.logical_or(terms, truncs).astype(np.float32), device=device)

                rewards_buf[step] = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            print(f"Update {update+1}/{num_updates} | Elapsed Time: {int(elapsed)}s | Steps: {global_step} | SPS: {int(sps)}")

            # Compute next values for GAE
            with torch.no_grad():
                next_values = []
                for env_i in range(num_envs):
                    s_i = next_obs[env_i].unsqueeze(0)
                    v_i = self.nnc(s_i).squeeze(-1)
                    next_values.append(v_i.item())
                next_values = torch.tensor(next_values, dtype=torch.float32, device=device)

                advantages, returns = self.compute_gae_vectorized(
                    rewards=rewards_buf,
                    values=values_buf,
                    dones=dones_buf,
                    next_done=next_done,
                    next_values=next_values
                )

            # PPO update
            self.learn_ppo_vectorized(
                obs_buf=obs_buf,
                actions_buf=actions_buf,
                logprob_buf=logprob_buf,
                values_buf=values_buf,
                advantages=advantages,
                returns=returns
            )

        print("[PPO] Training terminé.")
                
    def test_agent(self, env, testepisodes):
        total_rewards = []  # List to record the total rewards obtained by the agent
        if self.algo == "dqn":
            self.epsilon = 0  # Exploitation only during testing

        for episode in range(testepisodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
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
