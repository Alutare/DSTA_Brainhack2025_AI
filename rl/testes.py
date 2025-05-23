from safetensors.torch import load_file
from src.rl_manager import RLManager

if __name__ == "__main__":
    model_path = "./src/model/model.safetensors"
    state_dict = load_file(model_path)
    print("Keys in model.safetensors:")
    for key in state_dict.keys():
        print(key)

    manager = RLManager()
    obs = {
        "role": "scout",
        "observation": [0.0] * 40 
    }
    action = manager.rl(obs)
    print(f"Predicted action for role {obs['role']}: {action}")
