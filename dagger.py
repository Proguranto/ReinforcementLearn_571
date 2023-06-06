import torch
import torch.optim as optim
import numpy as np

from utils import rollout, relabel_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=50,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10):
    
    # TODO: Fill in your dagger implementation here. 
    
    # Hint: Loop through num_dagger_iters iterations, at each iteration train a policy on the current dataset.
    # Then rollout the policy, use relabel_action to relabel the actions along the trajectory with "expert_policy" and then add this to current dataset
    # Repeat this so the dataset grows with states drawn from the policy, and relabeled actions using the expert.
    
    # Optimizer code
    optimizer = optim.Adam(list(policy.parameters()))
    losses = []
    returns = []
    loss_fn = torch.nn.MSELoss()

    trajs = expert_paths

    # Flatten dataset
    dataset = {k: v for d in trajs for k, v in d.items()}
    for d in trajs:
        for k, v in d.items():
            if k not in dataset:
                dataset[k] = v
            else:
                dataset[k] = np.concatenate((dataset[k], v), axis=0)

    # Dagger iterations
    for dagger_itr in range(num_dagger_iters):
        idxs = np.array(range(len(trajs)))
        num_batches = len(idxs)*episode_length // batch_size
        losses = []

        # Train the model with Adam
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(num_batches):
                optimizer.zero_grad()
                # TODO start: Fill in your behavior cloning implementation 
                # Sample batches from data.
                obs = dataset["observations"]
                acts = dataset["actions"]
                sampled_indices = np.random.choice(a=range(obs.shape[0]), size=batch_size, replace=False)
                s_batch = np.array([obs[i] for i in sampled_indices])
                a_batch = np.array([acts[i] for i in sampled_indices])
                s_batch = torch.tensor(s_batch).to(torch.float32).to("cuda")
                a_batch = torch.tensor(a_batch).to(torch.float32).to("cuda")

                # Calculate new loss.
                a_hat = policy(s_batch)
                loss = loss_fn(a_hat, a_batch)

                # TODO end
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
            # print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss))
            losses.append(loss.item())

        # Collecting more data for dagger
        trajs_recent = []
        for k in range(num_trajs_per_dagger):
            env.reset()
            # TODO start: Rollout the policy on the environment to collect more data, relabel them, add them into trajs_recent
            rollouts = rollout(env=env, agent=policy, agent_name='dagger', episode_length=episode_length)
            relabelled_rollouts = relabel_action(rollouts, expert_policy)
            trajs_recent.append(relabelled_rollouts)
            # TODO end

        # Update dataset
        for dict in trajs_recent:
            for k, v in dict.items():
                if k not in dataset:
                    dataset[k] = v
                else:
                    dataset[k] = np.concatenate((dataset[k], v), axis=0)
        
        trajs += trajs_recent
        mean_return = np.mean(np.array([traj['rewards'].sum() for traj in trajs_recent]))
        print("Average DAgger return is " + str(mean_return))
        returns.append(mean_return)

    # print loss
    print("loss: ", losses)
    print("num_epochs: ", len(losses))