import os
import torch

def run_pipeline(env, agent, interface, mode: str):
    """
    - Training
    - Eventual Save
    - Tests
    - Generation of GIF
    - Close of env is there
    """
    if interface.didtrain:
        agent.train(env, interface.episodes)
        print(f"Entraînement terminé pour {interface.episodes} épisodes.")

        # Save (only ppo for now)
        if mode == "ppo":
            if interface.ask_save_ppo():
                os.makedirs(os.path.dirname(interface.path), exist_ok=True)
                torch.save(agent.nna.state_dict(), interface.path.replace(".pth", "_actor.pth"))
                torch.save(agent.nnc.state_dict(), interface.path.replace(".pth", "_critic.pth"))

    if interface.didtestfct():
        agent.test_agent(env, testepisodes=100)

    if interface.didgraphicfct():
        """interface.grahic_name = "assets/" 
        name = input("Enter the filename to save the graphic (without extension, default is cartpole.gif): ")
        if name.strip():
            interface.grahic_name += name + ".gif"
        else:
            interface.grahic_name += "cartpole.gif"

        agent.graphic_agent(filename=interface.grahic_name)
        print(f"Graphic saved as {interface.grahic_name}")"""
        print("Not implemented yet.")
        pass

    env.close()