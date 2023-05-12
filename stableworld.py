import torch
# import diffusers
# import library.model_util as model_util


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    print('Using GPU:', device)
else:
    print('Using CPU')

# Print PyTorch version
print('PyTorch version:', torch.__version__)