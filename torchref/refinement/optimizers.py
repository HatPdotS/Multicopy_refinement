import torch
from torch.optim import Adam

class AdamWithAdaptiveNoise(Adam):
    def __init__(self, params, lr=1e-3, alpha=0.1, eps=1e-8, update_weight=0.05,**kwargs):
        """
        Drop-in replacement for torch.optim.Adam with adaptive, scale-invariant noise injection.
        
        Args:
            params: model parameters
            lr: learning rate
            alpha: scaling factor for how much noise to inject per unit overfitting ratio
            eps: small constant for numerical stability
        """
        super().__init__(params, lr=lr, **kwargs)
        self.alpha = alpha
        self.eps = eps
        self.noise_scale = 0.0  # dynamically updated
        self.update_weight = update_weight

    @torch.no_grad()
    def inject_noise(self):
        """Inject scale-invariant Gaussian noise into gradients."""
        if self.noise_scale <= 0:
            return
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                gradnorm = torch.norm(grad)
                paramnorm = torch.norm(p)
                rms = paramnorm * 0.01 + gradnorm
                noise_std = self.noise_scale * self.alpha * rms
                grad.add_(torch.randn_like(grad) * noise_std)

    def step(self):
        """Perform a single optimization step with optional noise injection."""
        # Inject noise before the Adam update
        self.inject_noise()
        super().step()

    def update_noise_scale(self, train_nll, test_nll):
        """
        Update the noise scale based on the ratio of test to training NLL.
        Example: ratio > 1 means model is overfitting.
        """
        ratio = torch.log(torch.clamp(train_nll, min=1e-4)) - torch.log(torch.clamp(test_nll, min=1e-4))
        ratio = torch.clamp(ratio, min=0.0,max=0.1)  # only consider overfitting
        self.noise_scale = self.update_weight * ratio.item() + (1 - self.update_weight) * self.noise_scale
    
