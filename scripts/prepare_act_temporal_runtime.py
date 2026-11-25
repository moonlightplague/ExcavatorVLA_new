"""Copy LeRobot into a private runtime and extend only its ACT model/config.

Installed packages, evaluators, controllers and simulator files are untouched.
Use the emitted python launcher for both training and ordinary ACT evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Native ACT source changed; refusing an ambiguous runtime extension')
    return text.replace(old, new, 1)


def build(output, persistent=False):
    import lerobot
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = Path(lerobot.__file__).parent
    target = output / 'lerobot'
    shutil.copytree(package, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    config_path = target / 'policies/act/configuration_act.py'
    model_path = target / 'policies/act/modeling_act.py'
    original = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (config_path, model_path)}
    config = config_path.read_text()
    config = replace_once(config, '    n_obs_steps: int = 1\n',
        '    n_obs_steps: int = 1\n'
        '    temporal_horizon: int = 1\n'
        '    temporal_stride: int = 3\n'
        '    temporal_layers: int = 2\n'
        '    temporal_lr: float = 0.0001\n')
    config = replace_once(config, '        super().__post_init__()\n',
        '        super().__post_init__()\n'
        '        if self.temporal_horizon not in (1, 3):\n'
        '            raise ValueError("ACT temporal horizon must be one or three")\n'
        '        if self.temporal_horizon > 1 and (self.temporal_stride != self.n_action_steps\n'
        '                or self.temporal_layers < 1 or self.temporal_layers > 4\n'
        '                or self.temporal_lr <= 0 or self.temporal_ensemble_coeff is not None):\n'
        '            raise ValueError("Invalid internal ACT temporal configuration")\n')
    if persistent:
        config = replace_once(config, '    temporal_lr: float = 0.0001\n',
            '    temporal_lr: float = 0.0001\n'
            '    persistent_memory: bool = False\n'
            '    memory_hidden_dim: int = 256\n'
            '    memory_progress: bool = False\n'
            '    action_parameterization: str = "velocity"\n'
            '    command_pose_center: list[float] | None = None\n'
            '    command_pose_radius: list[float] | None = None\n'
            '    state_normalization_mean: list[float] | None = None\n'
            '    state_normalization_std: list[float] | None = None\n'
            '    action_normalization_mean: list[float] | None = None\n'
            '    action_normalization_std: list[float] | None = None\n')
        config = replace_once(config, '        super().__post_init__()\n',
            '        super().__post_init__()\n'
            '        if self.action_parameterization not in ("velocity", "command_pose"):\n'
            '            raise ValueError("Invalid action parameterization")\n'
            '        if self.action_parameterization == "command_pose" and not self.persistent_memory:\n'
            '            raise ValueError("Command-pose head requires persistent ACT")\n'
            '        if self.persistent_memory and (self.temporal_horizon != 3 or self.memory_hidden_dim < 16):\n'
            '            raise ValueError("Persistent ACT needs three-observation context and hidden_dim >= 16")\n')
    config_path.write_text(config)
    model = model_path.read_text()
    model = replace_once(model, '        self.model = ACT(config)\n',
        '        if config.temporal_horizon > 1:\n'
        '            from .temporal_extension import TemporalACT\n'
        '            self.model = TemporalACT(config)\n'
        '        else:\n'
        '            self.model = ACT(config)\n')
    anchor = '        # TODO(aliberts, rcadene): As of now, lr_backbone == lr\n'
    model = replace_once(model, anchor,
        '        if self.config.temporal_horizon > 1:\n'
        '            params = list(self.named_parameters())\n'
        '            return [\n'
        '                {"params": [p for n,p in params if p.requires_grad and not n.startswith(("model.backbone", "model.temporal_context"))]},\n'
        '                {"params": [p for n,p in params if p.requires_grad and n.startswith("model.backbone")], "lr": self.config.optimizer_lr_backbone},\n'
        '                {"params": [p for n,p in params if p.requires_grad and n.startswith("model.temporal_context")], "lr": self.config.temporal_lr},\n'
        '            ]\n' + anchor)
    reset_anchor = ('    def reset(self):\n'
                    '        """This should be called whenever the environment is reset."""\n')
    model = replace_once(model, reset_anchor, reset_anchor +
        '        if self.config.temporal_horizon > 1:\n'
        '            self.model.reset_history()\n')
    model = replace_once(model, '        actions = self.model(batch)[0]\n',
        '        actions = (self.model.forward_online(batch)[0] if self.config.temporal_horizon > 1\n'
        '                   else self.model(batch)[0])\n')
    if persistent:
        model = replace_once(model, '        if config.temporal_horizon > 1:\n',
            '        if config.persistent_memory:\n'
            '            from .persistent_extension import PersistentACT\n'
            '            self.model = PersistentACT(config)\n'
            '        elif config.temporal_horizon > 1:\n')
        model = replace_once(model, '    def get_optim_params(self) -> dict:\n',
            '    def _save_pretrained(self, save_directory):\n'
            '        if not self.config.persistent_memory:\n'
            '            return super()._save_pretrained(save_directory)\n'
            '        from safetensors.torch import save_file\n'
            '        self.config._save_pretrained(save_directory)\n'
            '        # cuDNN flattens GRU parameters into views of a shared allocation.\n'
            '        # Clone individual tensors to retain standard safetensors keys.\n'
            '        values = {k: v.detach().cpu().clone().contiguous() for k, v in self.state_dict().items()}\n'
            '        save_file(values, str(save_directory / "model.safetensors"))\n\n'
            '    def get_optim_params(self) -> dict:\n')
        model = replace_once(model,
            '            for img in batch[OBS_IMAGES]:\n'
            '                cam_features = self.backbone(img)["feature_map"]\n',
            '            for view, img in enumerate(batch[OBS_IMAGES]):\n'
            '                cam_features = (batch["_act_cached_backbone"][view] if "_act_cached_backbone" in batch\n'
            '                                else self.backbone(img)["feature_map"])\n')
    model_path.write_text(model)
    extension = target / 'policies/act/temporal_extension.py'
    shutil.copy2(BASE / 'ExcavatorVLA-vlm/excavator_vlm/policies/act/temporal.py', extension)
    for path in (config_path, model_path):
        compile(path.read_text(), str(path), 'exec')
    manifest = dict(version='act_internal_prediction_history_v1', source_package=str(package),
        interpreter=sys.executable, original_sha256=original,
        modified_sha256={str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (config_path, model_path, extension)},
        model_extension='lerobot.policies.act.temporal_extension',
        installed_packages_modified=False, evaluator_modified=False, controller_modified=False)
    if persistent:
        memory = target / 'policies/act/persistent_extension.py'
        shutil.copy2(BASE / 'ExcavatorVLA-vlm/excavator_vlm/policies/act/persistent.py', memory)
        compile(memory.read_text(), str(memory), 'exec')
        manifest['version'] = 'act_persistent_action_memory_v1'
        manifest['modified_sha256'][str(memory.relative_to(target))] = hashlib.sha256(memory.read_bytes()).hexdigest()
        pose = target / 'policies/act/command_pose_extension.py'
        shutil.copy2(BASE / 'ExcavatorVLA-vlm/excavator_vlm/policies/act/command_pose.py', pose)
        compile(pose.read_text(), str(pose), 'exec')
        manifest['modified_sha256'][str(pose.relative_to(target))] = hashlib.sha256(pose.read_bytes()).hexdigest()
        manifest['version'] = 'act_velocity_progress_v3'
    (output / 'runtime_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    launcher = output / 'python'
    path = str(output) + ':' + str(BASE / 'ExcavatorVLA-vlm')
    verifier = output / 'verify_runtime.py'
    verifier.write_text('import hashlib,json\nfrom pathlib import Path\n'
        'root=Path(__file__).resolve().parent\n'
        'manifest=json.loads((root/"runtime_manifest.json").read_text())\n'
        'for name,sha in manifest["modified_sha256"].items():\n'
        ' if hashlib.sha256((root/"lerobot"/name).read_bytes()).hexdigest()!=sha:\n'
        '  raise RuntimeError("ACT model runtime changed: "+name)\n')
    cudnn = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cudnn/lib'
    libraries = ('export LD_LIBRARY_PATH=' + shlex.quote(str(cudnn)) + ':"${LD_LIBRARY_PATH:-}"\n') if persistent else ''
    launcher.write_text('#!/bin/sh\n' + libraries + 'export PYTHONPATH=' + shlex.quote(path) + '\n'
                        'export PYTHONDONTWRITEBYTECODE=1\n' + shlex.quote(sys.executable) + ' -B '
                        + shlex.quote(str(verifier)) + ' || exit $?\nexec '
                        + shlex.quote(sys.executable) + ' "$@"\n')
    launcher.chmod(0o755)
    print(json.dumps(dict(runtime=str(output), python=str(launcher)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--persistent', action='store_true')
    args = parser.parse_args()
    build(args.output, persistent=args.persistent)
