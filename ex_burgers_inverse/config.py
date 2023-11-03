import torch
import os

########################################################################
# config.py
########################################################################
# For general settings
taskname = "burgers_inverse_task"
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # torch.float32 / torch.double

########################################################################
# For outputs settings
current_directory = os.path.dirname(os.path.realpath(__file__))
outputs_dir = "./outputs/"
dataset_path = os.path.join(current_directory, "dataset")
outputs_path = os.path.join(current_directory, outputs_dir)

########################################################################
# For training settings
learning_rate = 1e-3
iterations = 5000
LBFGS_iterations = 4000
print_every = 500

########################################################################
# For training settings
BackboneNet_input_dim = 2
BackboneNet_hidden_dim = 20
BackboneNet_output_dim = 1
BackboneNet_layers_num = 5






