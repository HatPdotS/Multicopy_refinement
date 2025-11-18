"""
LBFGS-based refinement framework for crystallographic structure refinement.

This module provides an LBFGS optimizer-based refinement approach which has been
shown to converge much faster than first-order optimizers (Adam, SGD, etc.).
LBFGS typically reaches near-convergence in just 1-2 macro cycles.
"""

import torch
from typing import Optional, Dict, List, Tuple
from torchref.refinement.base_refinement import Refinement


class LBFGSRefinement(Refinement):
    """
    LBFGS-based refinement subclass that uses the L-BFGS optimizer for fast convergence.
    
    L-BFGS (Limited-memory BFGS) is a quasi-Newton optimization method that approximates
    the Hessian matrix, leading to much faster convergence than first-order methods.
    
    Key advantages:
    - Converges in 1-2 macro cycles (vs 5+ for Adam)
    - Better final R-factors
    - More stable convergence
    - Automatically handles step size via line search
    
    Usage:
        refinement = LBFGSRefinement(mtz_file, pdb_file, 
                                      target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3})
        refinement.run_lbfgs_refinement(macro_cycles=2)
    """
    def grad_norm(self, loss):
        """Compute gradient norm with proper zeroing and graph cleanup"""

        # Zero gradients
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Collect gradients from parameters that have them
        grad_list = [p.grad.flatten().detach() for p in self.model.parameters() if p.grad is not None]
        
        # Check if we have any gradients
        if len(grad_list) == 0:
            # No parameters with gradients - return default value
            return 1.0
        
        # Concatenate and compute norm
        vec = torch.cat(grad_list)
        norm_value = vec.norm().item()
        
        # Clean up: delete gradient tensors and clear computation graph
        del vec
        del grad_list
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None
        
        return norm_value

    def refine_adp(self):
        """Refine B-factors (ADP)"""
        self.model.freeze_all()
        self.model.unfreeze('b')
        
        # Compute gradient-based weight using refinement.parameters() not model.parameters()
        loss_xray_ = self.xray_loss()
        loss_adp_ = self.adp_loss()

        gx = self.grad_norm(loss_xray_)
        ga = self.grad_norm(loss_adp_)
        weight_adp = (gx / (ga + 1e-12)) * self.target_weights['adp']
        self.effective_weights['adp'] = weight_adp

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.adp_loss() * self.effective_weights['adp'] + self.xray_loss()
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        self.model.unfreeze_all()

    def refine_xyz(self):
        """Refine coordinates (XYZ)"""
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze('xyz')

        # Compute gradient-based weight
        loss_xray_ = self.xray_loss()
        loss_geom_ = self.restraints_loss()

        gx = self.grad_norm(loss_xray_)
        gg = self.grad_norm(loss_geom_)

        weight_restraints = (gx / (gg + 1e-12)) * self.target_weights['restraints']
        self.effective_weights['restraints'] = weight_restraints

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.restraints_loss() * self.effective_weights['restraints'] + self.xray_loss()
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        self.model.unfreeze_all()
    
    def refine(self, macro_cycles=5):
        """Run full LBFGS refinement cycle (ADP + XYZ)"""
        
        self.scaler.freeze()
        i = 0

        while True:
            i += 1
            master_key = f'refinement_{i}'
            if not master_key in self.history:
                break

        self.history[master_key] = []
        for cycle in range(macro_cycles):
            cycle_dict = {}
            cycle_dict['cycle'] = cycle + 1
            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                cycle_dict['rwork_before_not_scaled'] = rwork
                cycle_dict['rfree_before_not_scaled'] = rfree
            self.get_scales()
            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                cycle_dict['rwork_before_scaled'] = rwork
                cycle_dict['rfree_before_scaled'] = rfree

            self.refine_xyz()
            with torch.no_grad():
                cycle_dict['rwork_after_xyz_scaled'], cycle_dict['rfree_after_xyz_scaled'] = self.get_rfactor()
                cycle_dict['restraints_weight'] = self.effective_weights['restraints']
                cycle_dict['nll_work_after_xyz'], cycle_dict['nll_free_after_xyz'] = self.nll_xray()
                cycle_dict['nll_bonds'] = self.restraints.nll_bonds().mean().item()
                cycle_dict['nll_angles'] = self.restraints.nll_angles().mean().item()
                cycle_dict['nll_torsion'] = self.restraints.nll_torsions().mean().item()
                cycle_dict['nll_planes'] = self.restraints.nll_planes().mean().item()
                cycle_dict['nll_vdw'] = self.restraints.nll_vdw().mean().item()

            self.refine_adp()
            with torch.no_grad():
                cycle_dict['rwork_after_adp_scaled'], cycle_dict['rfree_after_adp_scaled'] = self.get_rfactor()
                cycle_dict['adp_weight'] = self.effective_weights['adp']
                cycle_dict['nll_work_after_adp'], cycle_dict['nll_free_after_adp'] = self.nll_xray()

            for key in cycle_dict:
                if isinstance(cycle_dict[key], torch.Tensor):
                    cycle_dict[key] = cycle_dict[key].mean().item()



            self.history[master_key].append(cycle_dict)

        return self.history