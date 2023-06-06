import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NOTE: copy your implementation back to bc.py when you submit the code
def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, 
                       batch_size=32):
    
    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy. 
    optimizer = optim.Adam(list(policy.parameters()))
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs)*episode_length // batch_size
    losses = []
    loss_fn = torch.nn.MSELoss()
    dataset = {k: v for d in expert_data for k, v in d.items()}
    for d in expert_data:
        for k, v in d.items():
            if k not in dataset:
                dataset[k] = v
            else:
                dataset[k] = np.concatenate((dataset[k], v), axis=0)

    for epoch in range(num_epochs): 
        ## TODO Students
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here
            obs = dataset["observations"]
            acts = dataset["actions"]
            sampled_indices = np.random.choice(a=range(obs.shape[0]), size=batch_size, replace=False)
            s_batch = np.array([obs[i] for i in sampled_indices])
            a_batch = np.array([acts[i] for i in sampled_indices])
            s_batch = torch.tensor(s_batch).to(torch.float32).to("cuda")
            a_batch = torch.tensor(a_batch).to(torch.float32).to("cuda")
            a_hat = policy(s_batch)
            loss = loss_fn(a_hat, a_batch)
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 50 == 0:
            print('[%d] loss: %.8f' %
                (epoch, running_loss / 10.))
        losses.append(loss.item())
