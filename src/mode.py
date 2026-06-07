import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving differential equations.
    Designed for academic research presentation.
    """
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x, u_pred):
        """
        Placeholder for Physics-Informed Loss.
        In a real scenario, this function computes the residual of the PDE.
        """
        # Example: Simple derivative calculation for demonstration
        x.requires_grad_(True)
        u = self.forward(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # For demonstration, we assume a simple target like u_x = 0
        loss_pde = torch.mean(u_x**2)
        return loss_pde

# ==========================================
# Example Usage (For verification)
# ==========================================
if __name__ == "__main__":
    print("Initializing PINN model...")
    
    # 1. Create Model
    model = PINN(input_dim=1, hidden_dim=32, output_dim=1)
    
    # 2. Create dummy input data (x coordinates)
    x_train = torch.linspace(-1, 1, 10).view(-1, 1)
    
    # 3. Forward pass
    u_pred = model(x_train)
    
    # 4. Compute physics-informed loss
    loss = model.physics_loss(x_train, u_pred)
    
    print(f"Model successfully initialized.")
    print(f"Input shape: {x_train.shape}")
    print(f"Output shape: {u_pred.shape}")
    print(f"Initial Physics Loss: {loss.item():.6f}")
    print("\nStatus: Ready for training.")
