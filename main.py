from env.drone_env import DroneEnv
from core.interface import Interface
from core.runner import run_pipeline
from agents.ppo_agent import PPOAgent
from core.neural_network import NeuralNetwork
import torch

if __name__ == "__main__":
    interface = Interface()

    env = DroneEnv()
    n_actions = env.action_space.shape[0]

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
                print("DQN Model not find, lauch...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))

    run_pipeline(env=env, agent=agent, interface=interface, mode=mode)