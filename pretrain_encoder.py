import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from sprites_env.envs.sprites import *


def rollout_trajectory(env, num_steps = 50):
 
    states = []
    rewards = []

    state = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        states.append(next_state)
        rewards.append(reward)

        if done:
            break

    return np.array(states), np.array(rewards)


def train_model(env, model, num_trajectories = 5, num_steps = 40, input_steps = 3, future_steps = 20, epochs = 20, batch_size = 32, learning_rate = 0.005):
    """
    Efficiently train the RewardPredModel by extracting multiple samples from each trajectory.

    Parameters:
        env (SpritesEnv): The environment instance.
        model (RewardPredModel): The reward prediction model.
        num_trajectories (int): Total number of trajectories to generate.
        num_steps (int): Number of steps per trajectory.
        input_steps (int): Number of input steps for the model.
        future_steps (int): Number of future steps for reward prediction.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        losses (list): List of average losses per epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        samples = []
        targets = []

        for _ in range(num_trajectories):
            states, rewards = rollout_trajectory(env, num_steps)
            i = 0
            while i + input_steps + future_steps <= len(states):
                img_seq = torch.tensor(states[i:i + input_steps], dtype=torch.float32) 
                reward_sum = torch.tensor(rewards[i + input_steps : i + input_steps + future_steps].sum(), dtype=torch.float32) 
                samples.append(img_seq.unsqueeze(0))  # Add batch dimension
                targets.append(reward_sum.unsqueeze(0))
                i += 1  # Increment to move to the next sample


        # Shuffle data
        samples = torch.cat(samples, dim=0)  # Shape: (total_samples, input_steps, H, W)
        targets = torch.cat(targets, dim=0)  # Shape: (total_samples,)
        total_samples = samples.shape[0]
        perm = torch.randperm(total_samples)
        samples = samples[perm]
        targets = targets[perm]

        # Process in batches
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_samples = samples[batch_start:batch_end]
            batch_targets = targets[batch_start:batch_end]

            # Forward pass
            predicted_rewards = model(batch_samples).sum(dim=1)  # Predicted sum of rewards per sample

            # Compute loss
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


    env = SpritesEnv()
    model = RewardPredModel(input_channels=3, img_size=64, input_steps=3, output_steps=20)
    train_model(env, model, num_trajectories=50, num_steps=100, input_steps=3, future_steps=20, epochs=10, batch_size=32, learning_rate=0.01)
    torch.save(model.encoders.state_dict(), "encoders.pth")


 





