import os

class Interface:
    def __init__(self):
        self.didtrain = False
        self.episodes = None
        self.path_ppo = "tmp/model_saved/ppo/"
        self.grahic_name = None
        self.filename = None
        self.path = None

        # Create directories if they don't exist
        os.makedirs(self.path_ppo, exist_ok=True)

    def ask_mode(self):
        """user_input = input("Choose the agent mode - PPO or ... (p/...): ").lower()
        if user_input == 'p':
            return "ppo"
        else:   
            print("Invalid input, defaulting to DQN.")
            return "ppo" """
        return "ppo"

    def didtrainfct(self):
        user_input = input("Do you want to train the model? (y/n): ").lower()
        if user_input == 'y':
            self.didtrain = True
            self.episodes = int(input("How many episodes would you like to train the model for? "))

    def didtestfct(self):
        return input("Do you want to test the model? (y/n): ").lower() == 'y'

    def didgraphicfct(self):
        return input("Do you want to create a graphic of the agent's performance? (y/n): ").lower() == 'y'

    # ---------------------------
    # PPO
    # ---------------------------
    def ask_save_ppo(self):
        user_input = input("Do you want to save the PPO model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename to save the PPO model (without extension): ") + ".pth"
            self.path = os.path.join(self.path_ppo, self.filename)
            print(f"The PPO model will be saved at: {self.path}")
            return True
        return False
    
    def ask_load_ppo(self):
        user_input = input("Do you want to load an existing PPO model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename of the PPO model to load (without extension): ") + ".pth"
            self.path = os.path.join(self.path_ppo, self.filename)
            print(f"The PPO model will be loaded from: {self.path}")
            return True
        return False