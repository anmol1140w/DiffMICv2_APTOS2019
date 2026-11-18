"""Differentiable 2-Qubit & 4-Qubit Quantum Reservoir / Parametric Layers in Pure PyTorch.
- 100% GPU-native tensor operations (no CPU simulation bottleneck, no external C++ dependencies).
- Fully differentiable via PyTorch autograd.
- Vectorized Pauli single-qubit <Z_i> expectations and two-qubit <Z_i Z_j> correlators.
- Sub-millisecond execution per batch, enabling faster training with lower FLOPs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastQuantumLayer(nn.Module):
    """Parametric & Entangling Quantum Layer using 2 or 4 Qubits.
    Maps an input feature vector to quantum expectation observables.
    """
    def __init__(self, in_features: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        assert n_qubits in (2, 4), f"Supported qubits are 2 or 4, got {n_qubits}"
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        self.n_layers = n_layers
        
        # Classical -> Quantum angle projection
        self.proj = nn.Linear(in_features, n_qubits)
        # Variational parameters for post-entanglement rotations
        self.rot_y = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        
        # Precompute observable projection masks
        indices = torch.arange(self.dim)
        mask_1b = [torch.where((indices & (1 << q)) == 0, 1.0, -1.0) for q in range(n_qubits)]
        self.register_buffer('mask_1b', torch.stack(mask_1b, dim=1)) # (dim, n_qubits)
        
        pairs = []
        mask_2b = []
        for q1 in range(n_qubits):
            for q2 in range(q1 + 1, n_qubits):
                pairs.append((q1, q2))
                b1 = (indices & (1 << q1)) > 0
                b2 = (indices & (1 << q2)) > 0
                mask_2b.append(torch.where(b1 == b2, 1.0, -1.0))
        self.register_buffer('mask_2b', torch.stack(mask_2b, dim=1)) # (dim, num_pairs)
        self.out_dim = n_qubits + len(pairs) # 3 for 2Q, 10 for 4Q
        
        # Precompute CNOT permutations for circular ring entanglement
        self.cnot_perms = []
        for q in range(n_qubits):
            ctrl = q
            tgt = (q + 1) % n_qubits
            m = (indices & (1 << ctrl) != 0) & (indices & (1 << tgt) == 0)
            idx0 = indices[m]
            idx1 = idx0 | (1 << tgt)
            perm = indices.clone()
            perm[idx0] = idx1
            perm[idx1] = idx0
            perm_name = f'perm_{q}'
            self.register_buffer(perm_name, perm)
            self.cnot_perms.append(perm_name)

    def _apply_ry(self, state: torch.Tensor, theta: torch.Tensor, q: int) -> torch.Tensor:
        """Applies single-qubit Ry(theta) gate across batch via tensor reshaping."""
        B = state.shape[0]
        stride = 1 << q
        c = torch.cos(theta / 2.0).view(B, 1, 1, 1)
        s = torch.sin(theta / 2.0).view(B, 1, 1, 1)
        view = state.view(B, -1, 2, stride)
        s0 = view[:, :, 0:1, :]
        s1 = view[:, :, 1:2, :]
        new0 = c * s0 - s * s1
        new1 = s * s0 + c * s1
        return torch.cat([new0, new1], dim=2).view(B, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum circuit.
        Args:
            x: Tensor of shape (B, in_features)
        Returns:
            q_features: Tensor of shape (B, out_dim) where out_dim is 3 (2Q) or 10 (4Q)
        """
        B = x.shape[0]
        angles = torch.pi * torch.tanh(self.proj(x))
        
        # Initialize pure ground state |0...0>
        state = torch.zeros(B, self.dim, device=x.device, dtype=torch.float32)
        state[:, 0] = 1.0
        
        # Layer 1: Angle encoding Ry rotations
        for q in range(self.n_qubits):
            state = self._apply_ry(state, angles[:, q], q)
            
        # Layers 2..L: Entanglement and variational mixing
        for l in range(self.n_layers):
            for perm_name in self.cnot_perms:
                state = state[:, getattr(self, perm_name)]
            for q in range(self.n_qubits):
                state = self._apply_ry(state, self.rot_y[l, q].expand(B), q)
                
        # Compute projection probabilities and expectation observables
        probs = state.pow(2)
        exp_1b = torch.matmul(probs, self.mask_1b)
        exp_2b = torch.matmul(probs, self.mask_2b)
        return torch.cat([exp_1b, exp_2b], dim=-1)


class QuantumFusionSimplifier(nn.Module):
    """Quantum-Enhanced Feature Fusion for DCG.
    Replaces heavy MLP fusion branches with a 4-Qubit (or 2-Qubit) Quantum Circuit.
    """
    def __init__(self, in_features: int, num_classes: int = 5, n_qubits: int = 4):
        super().__init__()
        self.q_layer = FastQuantumLayer(in_features, n_qubits=n_qubits, n_layers=2)
        self.head = nn.Linear(self.q_layer.out_dim, num_classes, bias=False)

    def forward(self, h_global: torch.Tensor, z_local: torch.Tensor):
        concat = torch.cat([h_global, z_local], dim=-1)
        q_features = self.q_layer(concat)
        y_quantum = self.head(q_features)
        return y_quantum, q_features


class QuantumConditioningBridge(nn.Module):
    """Quantum Cross-Conditioning Bridge for DiffMICv2 Diffusion Models.
    Entangles generative diffusion noisy labels with visual disease representations
    across a 4-qubit (or 2-qubit) Hilbert space, generating FiLM modulation parameters.
    """
    def __init__(self, feature_dim: int, n_qubits: int = 4):
        super().__init__()
        assert n_qubits in (2, 4)
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        
        q_half = n_qubits // 2
        self.proj_y = nn.Linear(feature_dim, q_half)
        self.proj_x = nn.Linear(feature_dim, q_half)
        self.rot_y = nn.Parameter(torch.randn(2, n_qubits) * 0.1)
        
        indices = torch.arange(self.dim)
        # Precompute cross-subspace CNOT entangling gates
        self.cnot_perms = []
        cnot_pairs = [(0, 2), (1, 3), (2, 0), (3, 1)] if n_qubits == 4 else [(0, 1), (1, 0)]
        for idx, (ctrl, tgt) in enumerate(cnot_pairs):
            m = (indices & (1 << ctrl) != 0) & (indices & (1 << tgt) == 0)
            idx0 = indices[m]
            idx1 = idx0 | (1 << tgt)
            perm = indices.clone()
            perm[idx0] = idx1
            perm[idx1] = idx0
            perm_name = f'perm_{idx}'
            self.register_buffer(perm_name, perm)
            self.cnot_perms.append(perm_name)
            
        # Correlator measurement masks
        pairs = [(0, 2), (1, 3)] if n_qubits == 4 else [(0, 1)]
        mask_2b = []
        for q1, q2 in pairs:
            b1 = (indices & (1 << q1)) > 0
            b2 = (indices & (1 << q2)) > 0
            mask_2b.append(torch.where(b1 == b2, 1.0, -1.0))
        self.register_buffer('mask_2b', torch.stack(mask_2b, dim=1))
        
        mask_1b = [torch.where((indices & (1 << q)) == 0, 1.0, -1.0) for q in range(n_qubits)]
        self.register_buffer('mask_1b', torch.stack(mask_1b, dim=1))
        
        num_obs = n_qubits + len(pairs)
        self.film_gen = nn.Linear(num_obs, feature_dim * 2)

    def _apply_ry(self, state: torch.Tensor, theta: torch.Tensor, q: int) -> torch.Tensor:
        B = state.shape[0]
        stride = 1 << q
        c = torch.cos(theta / 2.0).view(B, 1, 1, 1)
        s = torch.sin(theta / 2.0).view(B, 1, 1, 1)
        view = state.view(B, -1, 2, stride)
        s0 = view[:, :, 0:1, :]
        s1 = view[:, :, 1:2, :]
        new0 = c * s0 - s * s1
        new1 = s * s0 + c * s1
        return torch.cat([new0, new1], dim=2).view(B, self.dim)

    def forward(self, y_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Modulates diffusion features using quantum cross-subspace entanglement.
        Args:
            y_feat: Diffusion tensor (B, feature_dim, H, W)
            x_feat: Visual condition tensor (B, feature_dim)
        Returns:
            Modulated diffusion feature tensor (B, feature_dim, H, W)
        """
        B = x_feat.shape[0]
        y_pool = y_feat.mean(dim=[2, 3]) if y_feat.dim() == 4 else y_feat
        
        angles_y = torch.pi * torch.tanh(self.proj_y(y_pool))
        angles_x = torch.pi * torch.tanh(self.proj_x(x_feat))
        angles = torch.cat([angles_y, angles_x], dim=-1)
        
        state = torch.zeros(B, self.dim, device=x_feat.device, dtype=torch.float32)
        state[:, 0] = 1.0
        
        for q in range(self.n_qubits):
            state = self._apply_ry(state, angles[:, q], q)
            
        for perm_name in self.cnot_perms:
            state = state[:, getattr(self, perm_name)]
            
        for q in range(self.n_qubits):
            state = self._apply_ry(state, self.rot_y[0, q].expand(B), q)
            
        probs = state.pow(2)
        exp_1b = torch.matmul(probs, self.mask_1b)
        exp_2b = torch.matmul(probs, self.mask_2b)
        q_obs = torch.cat([exp_1b, exp_2b], dim=-1)
        
        film = self.film_gen(q_obs)
        gamma, beta = film.chunk(2, dim=-1)
        if y_feat.dim() == 4:
            gamma = gamma.view(B, -1, 1, 1)
            beta = beta.view(B, -1, 1, 1)
        return (1.0 + gamma) * y_feat + beta
