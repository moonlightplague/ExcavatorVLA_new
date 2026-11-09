"""Alternative ACT reconstruction loss with original PHW/padding/KL semantics."""
import math
import torch
import torch.distributed as dist
from torch.nn import functional as F


def reconstruction_loss(prediction, target, padding, supervision, kind='l1', beta=.1, executed_steps=3):
    from excavator_vlm.policies.act.adapter import weighted_l1

    original, weights = weighted_l1(prediction, target, padding, supervision)
    if kind == 'l1':
        return original, weights, original
    if kind not in ('smooth_l1','executed_prefix_l1') or isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError('Expected l1 or smooth_l1 with a positive finite beta')
    element = F.smooth_l1_loss(prediction.float(), target.float(), reduction='none', beta=beta)
    if kind == 'executed_prefix_l1':
        if type(executed_steps) is not int or not 1 <= executed_steps <= prediction.shape[1]:
            raise ValueError('Executed prefix must lie inside the original action chunk')
        prefix = torch.arange(prediction.shape[1],device=prediction.device)[None,:,None] < executed_steps
        element = torch.where(prefix,(prediction.float()-target.float()).abs(),element)
    denominator = (weights.sum() * target.shape[-1]).detach().clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(denominator)
        world = dist.get_world_size()
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    loss = (element * weights.unsqueeze(-1)).sum() * world / safe_denominator
    return loss, weights, original


def policy_objective(policy, batch, kind='l1', beta=.1):
    if policy.config.image_features:
        batch = {**batch, 'observation.images': [batch[key] for key in policy.config.image_features]}
    predicted, (mu, logvar) = policy.model(batch)
    reconstruction, weights, original = reconstruction_loss(predicted, batch['action'],
        batch['action_is_pad'], batch['action_loss_weight'], kind, beta,
        executed_steps=getattr(policy.config,'n_action_steps',3))
    kl = reconstruction.new_zeros(())
    if policy.config.use_vae:
        active = (weights.sum(-1) > 0).float()
        count = active.sum().detach().clone()
        world = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            world = dist.get_world_size()
        per_sample = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1)
        kl = (per_sample * active).sum() * world / count.clamp_min(1)
    loss = reconstruction + policy.config.kl_weight * kl
    return loss, dict(reconstruction_loss=float(reconstruction.detach()),
        original_l1_diagnostic=float(original.detach()), kld_loss=float(kl.detach()),
        total_loss=float(loss.detach()))
