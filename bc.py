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

    observations = torch.tensor([])
    actions = torch.tensor([])
    for i in idxs:
      o = torch.tensor(expert_data[i]["observations"])
      a = torch.tensor(expert_data[i]["actions"])
      observations = torch.cat((observations, o), 0)
      actions = torch.cat((actions, a), 0)

    observations = observations.to(torch.float32).to("cuda")
    actions = actions.to(torch.float32).to("cuda")

    for epoch in range(num_epochs): 
        ## TODO Students
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here
            s_batch, a_batch = observations[i * batch_size : (i + 1) * batch_size], actions[i * batch_size : (i + 1) * batch_size]
            a_hat = policy.forward(s_batch)
            loss = torch.nn.MSELoss()(a_hat, a_batch)
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 50 == 0:
            print('[%d] loss: %.8f' %
                (epoch, running_loss / 10.))
        losses.append(loss.item())
