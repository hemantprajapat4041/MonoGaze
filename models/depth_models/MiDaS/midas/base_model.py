import torch


class BaseModel(torch.nn.Module):
<<<<<<< HEAD
   # In base_model.py, around line 16, change:
    def load(self, path):
        """Load model from file."""
        parameters = torch.load(path, map_location=torch.device('cpu'))
        # Change from:
        self.load_state_dict(parameters)
        # To:
        self.load_state_dict(parameters, strict=False)

=======
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
>>>>>>> origin/main
