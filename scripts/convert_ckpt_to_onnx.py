import argparse
import os
import sys

import isaacgym  # must import before torch

import yaml
import torch

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(LEGGED_GYM_ROOT_DIR)

from rsl_rl.modules import ActorCriticCTS

# import exporter directly to avoid circular imports via legged_gym.__init__
sys.path.append(os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'utils'))
import importlib
exporter = importlib.import_module('exporter')
export_policy_as_onnx = exporter.export_policy_as_onnx
export_policy_as_jit = exporter.export_policy_as_jit


def build_model(cfg, device='cpu'):
    train_cfg = cfg['train_cfg']
    env_cfg = cfg['env_cfg']
    policy_cfg = train_cfg['policy']

    model = ActorCriticCTS(
        num_actor_obs=env_cfg['env']['num_observations'],
        num_critic_obs=env_cfg['env']['num_privileged_obs'],
        num_actions=env_cfg['env']['num_actions'],
        num_envs=1,
        history_length=train_cfg['history_length'],
        actor_hidden_dims=policy_cfg['actor_hidden_dims'],
        critic_hidden_dims=policy_cfg['critic_hidden_dims'],
        student_encoder_hidden_dims=policy_cfg['student_encoder_hidden_dims'],
        teacher_encoder_hidden_dims=policy_cfg['teacher_encoder_hidden_dims'],
        activation=policy_cfg['activation'],
        init_noise_std=policy_cfg['init_noise_std'],
        latent_dim=policy_cfg['latent_dim'],
        norm_type=policy_cfg['norm_type'],
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model_xx.pt')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--output', type=str, default=None, help='Output onnx path')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # locate config
    config_path = args.config or os.path.join(checkpoint_dir, 'config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(checkpoint_dir), 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Building model...")
    model = build_model(cfg, args.device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # output path
    if args.output:
        output_dir = os.path.dirname(args.output)
        filename = os.path.basename(args.output)
    else:
        output_dir = os.path.join(checkpoint_dir, 'exported', 'policies')
        filename = 'policy.onnx'
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, filename)
    print(f"Exporting ONNX to: {onnx_path}")
    export_policy_as_onnx(model, output_dir, filename=filename, verbose=True)

    # also export JIT .pt for convenience
    jit_path = os.path.join(output_dir, filename.replace('.onnx', '.pt'))
    print(f"Exporting JIT to: {jit_path}")
    export_policy_as_jit(model, output_dir, filename=filename.replace('.onnx', '.pt'))

    print("Done!")


if __name__ == '__main__':
    main()
