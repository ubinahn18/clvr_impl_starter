import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from sprites_env.envs.sprites import *


def rollout_trajectory(env, num_steps = 40):
 
    states = []
    imgs = []

    state = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_img, next_state, _, done, _ = env.step(action)

        states.append(next_state)
        imgs.append(next_img)

        if done:
            break

    return np.array(states), np.array(imgs)



def traj_rewards(traj, reward_type):
    import numpy as np
    
    # Define valid reward types and their corresponding indices
    valid_reward_types = {
        "AgentXReward": (0, 1),
        "AgentYReward": (0, 0),
        "TargetXReward": (1, 0),
        "TargetYReward": (1, 1),
    }

    # Ensure reward_type is a list
    if isinstance(reward_type, str):
        reward_type = [reward_type]
    
    # Validate each reward type in the list
    for rt in reward_type:
        assert rt in valid_reward_types, f"Invalid reward_type: {rt}. Must be one of {set(valid_reward_types.keys())}."
    
    # Extract rewards based on reward_type and stack them
    rewards = [traj[:, indices[0], indices[1]] for rt in reward_type for indices in [valid_reward_types[rt]]]
    return np.stack(rewards, axis=-1)

 


def train_model(env, model, num_trajectories = 5, num_steps = 40, input_steps = 3, future_steps = 20, epochs = 20, batch_size = 32, learning_rate = 0.005):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    reward_type = ["AgentXReward", "AgentYReward", "TargetXReward", "TargetYReward"]
 
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        samples = []
        targets = []

        for _ in range(num_trajectories):
            states, imgs = rollout_trajectory(env, num_steps)
            rewards = traj_rewards(states, reward_type)
            i = 0
            while i + input_steps + future_steps <= len(states):
                img_seq = torch.tensor(imgs[i:i + input_steps], dtype=torch.float32) 
                future_reward_seq = torch.tensor(rewards[i + input_steps : i + input_steps + future_steps], dtype=torch.float32) 
                # unsqueeze to add batch dimension
                samples.append(img_seq.unsqueeze(0)) 
                targets.append(rewards.unsqueeze(0))
                i += 1


        samples = torch.cat(samples, dim=0)  # Shape: (total_samples, input_steps, H, W)
        targets = torch.cat(targets, dim=0)  # Shape: (total_samples,)
        total_samples = samples.shape[0]
        perm = torch.randperm(total_samples)
        samples = samples[perm]
        targets = targets[perm]

        # Process in batches
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_samples = samples[batch_start:batch_end,:]
            # print(batch_samples.shape)
            batch_targets = targets[batch_start:batch_end,:]

            predicted_rewards = model(batch_samples)

            loss = criterion(predicted_rewards, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (batch_end - batch_start)

        avg_loss = epoch_loss / total_samples
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return losses

if __name__ == "__main__":

    data_spec = AttrDict(
        resolution=64,
        max_ep_len=40,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True,
    )
    reward_type = ["AgentXReward", "AgentYReward", "TargetXReward", "TargetYReward"]

    env = SpritesStateImgEnv()
    model = RewardPredModel(reward_type, input_channels=1, img_size=64, input_steps=3, output_steps=20)
    train_model(env, model, num_trajectories=50, num_steps=100, input_steps=3, future_steps=20, epochs=10, batch_size=32, learning_rate=0.01)
    torch.save(model.encoders.state_dict(), "encoders.pth")


 





