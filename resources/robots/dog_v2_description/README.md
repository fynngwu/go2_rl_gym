# Dog V2 Description

URDF and MJCF model files for the Dog V2 quadruped robot.

## Usage

Training uses `urdf/dog_v2_2_4.urdf` as the primary model.

```python
URDF_PATH = "urdf/dog_v2_2_4.urdf"
```

## File Structure

```
dog_v2_description/
├── README.md
├── dog_v2_2_4.xml          # MJCF model
├── meshes/                  # STL mesh files
│   ├── base_link.STL
│   ├── LF_*.STL
│   ├── RF_*.STL
│   ├── LR_*.STL
│   └── RR_*.STL
└── urdf/
    ├── dog_v2_2_4.urdf      # Primary URDF (for training)
    └── backup/
        └── dog_v2.urdf      # Legacy URDF (unused)
```

## Models

| File | Description |
|------|-------------|
| `urdf/dog_v2_2_4.urdf` | Primary URDF with simplified collision geometry, used for RL training |
| `urdf/backup/dog_v2.urdf` | Legacy raw export from SolidWorks, mesh-based collisions |
| `dog_v2_2_4.xml` | MuJoCo MJCF format model |
