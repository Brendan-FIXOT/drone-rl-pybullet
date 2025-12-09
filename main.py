import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from env.drone_env import DroneEnv
from core.interface import Interface
from core.runner import run_pipeline
from agents.ppo_agent import PPOAgent
from core.neural_network import NeuralNetwork
from env.vecto_env import create_vector_async_env
from env.vecto_env import create_vector_sync_env

if __name__ == "__main__":
    interface = Interface()

    try :
        num_envs = int(input("How many env ? (1 for single env, more for vectorized envs) : "))
    except ValueError:
        print("Invalid input, defaulting to 1 environment.")
        num_envs = 1

    if num_envs > 1:
        async_choice = int(input("Choose vectorized environment type: 1 for AsyncVectorEnv, 2 for SyncVectorEnv: "))
        if async_choice == 1:
            env = create_vector_async_env(num_envs, seed=42)
        else:
            env = create_vector_sync_env(num_envs, seed=42)
        print("Vector envs créés avec", env.num_envs, "environnements.")
    else:
        env = DroneEnv()

    n_actions = 4
    
    mode = interface.ask_mode()  # only ppo for now

    if mode == "ppo":
        
        actor = NeuralNetwork(
            input_dim=12,
            output_dim=n_actions,
            mode="actor",
            lr=3e-4
        )
        critic = NeuralNetwork(
            input_dim=12,
            output_dim=1,
            mode="critic",
            lr=1e-3
        )

        agent = PPOAgent(
            actor_nn=actor,
            critic_nn=critic,
            n_actions=n_actions,
            buffer_size=512,
            batch_size=64,
            nb_epochs=4,
            entropy_bonus=False
        )

        if interface.ask_load_ppo():
            try:
                agent.nna.load_state_dict(torch.load(interface.path.replace(".pth", "_actor.pth")))
                agent.nnc.load_state_dict(torch.load(interface.path.replace(".pth", "_critic.pth")))
                print("Model PPO find.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("PPO Model not find, lauch...")
                interface.didtrain = True
                if num_envs == 1 : interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            if num_envs == 1 : interface.episodes = int(input("How many episodes would you like to train the model for? "))

    run_pipeline(env=env, num_envs=num_envs, agent=agent, interface=interface, mode=mode)