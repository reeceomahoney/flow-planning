import time

from flow_planning.particle import ParticleEnv

device = "cuda"
save_path = "data/flow_planning/particle_env/dataset.pt"
env = ParticleEnv(
    num_envs=10000,
    grid_size=1.0,
    process_noise=0.02,
    measurement_noise=0.01,
    init_pos_var=0.05,
    kp=1.0,
    kd=0.5,
    dt=0.05,
    seed=42,
    device=device,
)

start_time = time.time()
dataset = env.generate_dataset(
    num_samples=10000,
    trajectory_length=64,
    save_path=save_path,
)
print(
    f"Generated dataset with obs shape {dataset['obs'].shape} samples in"
    f" {time.time() - start_time:.2f} seconds"
)
print(f"Saved dataset to {save_path}")
env.visualize_trajectories(dataset, batch_size=50)
