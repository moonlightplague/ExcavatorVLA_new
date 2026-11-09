"""Training-only stage supervision from ACT's current observation features."""
import torch
from torch import nn
from torch.nn import functional as F


def stage_labels(batch, by_identity, training_episodes):
    """Join labels by episode/frame, rejecting held-out or unknown observations."""
    allowed = set(training_episodes)
    labels = []
    episodes = batch['episode_index'].reshape(-1).tolist()
    frames = batch['frame_index'].reshape(-1).tolist()
    for episode, frame in zip(episodes, frames, strict=True):
        key = (int(episode), int(frame))
        if key != (episode, frame) or key[0] not in allowed or key not in by_identity:
            raise ValueError(f'Unknown or held-out stage observation: {episode}/{frame}')
        stage = by_identity[key]
        if type(stage) is not int or not 0 <= stage < 10:
            raise ValueError('Expected an original current-stage label in [0, 9]')
        labels.append(stage)
    return torch.tensor(labels, dtype=torch.long)


class CurrentObservationFeatures:
    """Capture state/image projections before attention can mix in the VAE latent.

    Encoder/decoder outputs in ACT's training path can see future actions through
    the posterior latent. This auxiliary task deliberately uses neither output.
    The context manager does not register new modules on the ACT policy.
    """
    def __init__(self, policy):
        self.model = policy.model
        self.cameras = len(policy.config.image_features)
        self.handles = []
        self.reset()

    def reset(self):
        self.state = None
        self.images = []

    def __enter__(self):
        def state_hook(module, inputs, output):
            if self.state is not None:
                raise RuntimeError('Reset captured features before each policy forward')
            self.state = output

        def image_hook(module, inputs, output):
            self.images.append(output)

        self.handles = [self.model.encoder_robot_state_input_proj.register_forward_hook(state_hook)]
        if self.cameras:
            self.handles.append(self.model.encoder_img_feat_input_proj.register_forward_hook(image_hook))
        return self

    def tensor(self):
        if self.state is None or len(self.images) != self.cameras:
            raise RuntimeError('Incomplete current-observation feature capture')
        if self.state.ndim != 2 or any(x.ndim != 4 or x.shape[:2] != self.state.shape for x in self.images):
            raise ValueError('Unexpected ACT state/image projection shapes')
        return torch.cat([self.state.float(), *(x.float().mean((-2, -1)) for x in self.images)], dim=-1)

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.reset()


class StageAuxiliaryHead(nn.Module):
    def __init__(self, hidden_dim, cameras):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * (1 + cameras))
        self.classifier = nn.Linear(hidden_dim * (1 + cameras), 10)

    def forward(self, observation_features):
        return self.classifier(self.norm(observation_features))


def auxiliary_loss(head, captured, labels, class_weights):
    logits = head(captured.tensor())
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError('Stage labels must describe current observations')
    if class_weights.shape != (10,) or not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
        raise ValueError('Expected ten finite positive training-only class weights')
    loss = F.cross_entropy(logits.float(), labels, weight=class_weights.float())
    return loss, logits
