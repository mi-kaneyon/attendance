import torch

def load_and_analyze_model(model_path):
    # Load the state_dict from the saved model file
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Print out the keys (layer names) in the state_dict
    for key in state_dict.keys():
        print(key)

if __name__ == '__main__':
    model_path = 'script/trained_model.pth'  # Replace with the actual path to your save_model.pth file
    load_and_analyze_model(model_path)
