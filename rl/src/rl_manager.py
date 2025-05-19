import torch
import os
import numpy as np
from safetensors.torch import load_file
from model5.dqn_model import DQN
import logging
logging.basicConfig(level=logging.INFO)

class RLManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use input dimension of 43 to match the saved model
        input_dim = 43
        output_dim = 5 
        
        model_path = os.path.join(os.path.dirname(__file__), "model5", "model.safetensors")
        full_state_dict = load_file(model_path, device=str(self.device))
        
        self.models = {}
        for role in ["scout", "guard"]:
            model = DQN(input_dim, output_dim).to(self.device)
            
            role_state_dict = {
                k.replace(f"{role}_model.", ""): v
                for k, v in full_state_dict.items()
                if k.startswith(f"{role}_model.")
            }
            model.load_state_dict(role_state_dict)
            model.eval()
            self.models[role] = model
            logging.info(f"Loaded {role} model successfully")
    
    def flatten_obs(self, obs_dict):
        """
        Convert observation to a flat vector with the original format (43 elements)
        to match the saved model's expectations.
        """
        # Flatten viewcone (7x5=35 elements)
        flat_view = np.array(obs_dict["viewcone"]).flatten() / 255.0
        
        # Add direction (one-hot encoding, 4 elements)
        direction_onehot = np.zeros(4)
        direction_onehot[obs_dict["direction"]] = 1
        
        # Add scout/guard indicator (1 element)
        is_scout = np.array([obs_dict["scout"]])
        
        # Add location (2 elements)
        location = np.array(obs_dict["location"]) / 15.0  # Normalize to [0,1]
        
        # Add step count (1 element) - using the original format, not the sinusoidal encoding
        step = np.array([obs_dict["step"] / 100.0])
        
        # Combine all features (total: 35 + 4 + 1 + 2 + 1 = 43 elements)
        return np.concatenate([flat_view, direction_onehot, is_scout, location, step])
    
    def rl(self, observation: dict) -> int:
        logging.info(f"Received observation: {observation}")
        
        # Extract the inner observation dictionary
        if "observation" in observation:
            obs_dict = observation["observation"]
        else:
            obs_dict = observation
        
        # Determine role (scout/guard)
        role = "scout" if obs_dict.get("scout", 0) == 1 else "guard"
        if role not in self.models:
            raise ValueError(f"Unknown role '{role}'")
        
        # Use the original flattening function matching the model dimensions
        obs_array = self.flatten_obs(obs_dict)
        logging.info(f"Processed observation array shape: {obs_array.shape}")
        
        # Convert to tensor and get model prediction
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.models[role](obs_tensor)
            logging.info(f"Q-values: {q_values}")
            action = torch.argmax(q_values, dim=1).item()
            logging.info(f"Chosen action: {action}")
        
        return action