import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define the dataset points2D class unconditional
class Points2Dataset_uncond(Dataset):

    def __init__(self, points_count):
        self.points_count = points_count
        self.points2D = self.create_points()

    def __len__(self):
        return self.points_count

    def __getitem__(self, idx):
        return self.points2D[idx]

    def create_points(self):
        return np.random.uniform(low=-1, high=1, size=(self.points_count, 2))


# Define the dataset points2D class conditional
class Points2Dataset_cond(Dataset):

    def __init__(self, points_count, c):
        self.points_count = points_count
        self.c = c
        self.points2D = self.create_points()

    def __len__(self):
        return self.points_count

    def __getitem__(self, idx):
        return self.points2D[idx]

    def create_points(self):
        if self.c == 0:
            x = np.random.uniform(low=-1, high=-0.6, size=self.points_count)
            y = np.random.uniform(low=-1, high=1, size=self.points_count)
            return np.array([x, y]).T

        if self.c == 1:
            x = np.random.uniform(low=-0.6, high=-0.2, size=self.points_count)
            y = np.random.uniform(low=-1, high=1, size=self.points_count)
            return np.array([x, y]).T
        if self.c == 2:
            x = np.random.uniform(low=-0.2, high=0.2, size=self.points_count)
            y = np.random.uniform(low=-1, high=1, size=self.points_count)
            return np.array([x, y]).T
        if self.c == 3:
            x = np.random.uniform(low=0.2, high=0.6, size=self.points_count)
            y = np.random.uniform(low=-1, high=1, size=self.points_count)
            return np.array([x, y]).T
        else:
            x = np.random.uniform(low=0.6, high=1, size=self.points_count)
            y = np.random.uniform(low=-1, high=1, size=self.points_count)
            return np.array([x, y]).T


#  create a color square
def print_class_c():
    # if c == 0:
    plt.plot([-1, -0.6], [-1, -1], color='red')
    plt.plot([-0.6, -0.6], [-1, 1], color='red')
    plt.plot([-0.6, -1], [1, 1], color='red')
    plt.plot([-1, -1], [1, -1], color='red')
    # if c == 1:
    plt.plot([-0.6, -0.2], [-1, -1], color='yellow')
    plt.plot([-0.2, -0.2], [-1, 1], color='yellow')
    plt.plot([-0.2, -0.6], [1, 1], color='yellow')
    plt.plot([-0.6, -0.6], [1, -1], color='yellow')
    # if c == 2:
    plt.plot([-0.2, 0.2], [-1, -1], color='orange')
    plt.plot([0.2, 0.2], [-1, 1], color='orange')
    plt.plot([0.2, -0.2], [1, 1], color='orange')
    plt.plot([-0.2, -0.2], [1, -1], color='orange')
    # if c == 3:
    plt.plot([0.2, 0.6], [-1, -1], color='green')
    plt.plot([0.6, 0.6], [-1, 1], color='green')
    plt.plot([0.6, 0.2], [1, 1], color='green')
    plt.plot([0.2, 0.2], [1, -1], color='green')
    # if c == 4:
    plt.plot([0.6, 1], [-1, -1], color='blue')
    plt.plot([1, 1], [-1, 1], color='blue')
    plt.plot([1, 0.6], [1, 1], color='blue')
    plt.plot([0.6, 0.6], [1, -1], color='blue')


# Given an array of points returns an array of the classes matched to each point accordingly
def create_colors_array(points):

    colors = []
    for point in points:
        p = point[0].detach()
        if p < -0.6:
            colors.append(0)
            continue
        if p < -0.2:
            colors.append(1)
            continue
        if p < 0.2:
            colors.append(2)
            continue
        if p < 0.6:
            colors.append(3)
            continue
        else:
            colors.append(4)
            continue
    return colors


# Define the diffusion model architecture
class DiffusionModel_cond(nn.Module):
    def __init__(self):
        super(DiffusionModel_cond, self).__init__()
        self.embedding = torch.nn.Embedding(5, 2)
        self.fc1 = nn.Linear(5, 100)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x, t, c):
        x_t = torch.cat((torch.tensor(x), torch.tensor(t)), dim=1).float()
        c = self.embedding(torch.tensor(c)).squeeze(1)
        x_t_c = torch.cat((x_t, c), dim=1).float()
        x_t_c = self.fc1(x_t_c)
        x_t_c = self.relu1(x_t_c)
        x_t_c = self.fc2(x_t_c)
        x_t_c = self.relu2(x_t_c)
        x_t_c = self.fc3(x_t_c)
        return x_t_c


# Define the diffusion model architecture
class DiffusionModel_uncond(nn.Module):
    def __init__(self):
        super(DiffusionModel_uncond, self).__init__()
        self.fc1 = nn.Linear(3, 100)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x, t):
        x = torch.cat((torch.tensor(x), torch.tensor(t)), dim=1).float()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# The following function performs the reverse process
def reverse_process_DDIM_uncond(denoiser, dt, flag_Noise):
    factor = torch.randn(1, 2) * 0.01
    if flag_Noise is not None:
        z = flag_Noise
    else:
        z = torch.randn(1000, 2)
    z.requires_grad = True
    t = 1
    z_process = []
    while t >= 0:
        if flag_Noise is not None:
            time = torch.tensor([[t]])
        else:
            time = torch.full((number_of_points,), t).unsqueeze(1)
        epsilon = denoiser(z, time)
        sigma = np.exp(5 * (t - 1))
        x_t = z + factor - epsilon * sigma
        score = (x_t - z) / sigma ** 2
        dz = 5 * sigma ** 2 * dt * score
        z = z + dz
        t -= dt
        z_process.append(z.detach().numpy())
    return z.detach().numpy().squeeze(), np.array(z_process).squeeze()


# The following function performs the reverse process
def reverse_process_DDIM_cond(denoiser, dt, flag_Noise,c):
    if flag_Noise is not None:
        z = flag_Noise
    else:
        z = torch.randn(1000, 2)
    z.requires_grad = True
    t = 1
    z_process = []
    while t >= 0:
        if flag_Noise is not None:
            time = torch.tensor([[t]])
        else:
            time = torch.full((number_of_points,), t).unsqueeze(1)
            c = np.repeat(np.arange(0, 5), 200)
        epsilon = denoiser(z, time, c)
        sigma = np.exp(5 * (t - 1))
        x_t = z - epsilon * sigma
        score = (x_t - z) / sigma ** 2
        dz = 5 * sigma ** 2 * dt * score
        z = z + dz
        t -= dt
        z_process.append(z.detach().numpy())
    return z.detach().numpy().squeeze(), np.array(z_process).squeeze()

def snr(t):
    return 1 / np.exp(5 * (t - 1)) ** 2


# Estimate the probability for each point
def calc_p(point, t, D, c):
    T = 1000
    dt = 1 / T
    diff = 0
    sigma = np.exp(5 * (t - 1))

    for i in range(2000):
        epsilon = torch.randn(1, 2)
        x_t = point + sigma * epsilon
        estimated_noise = D(x_t, t, c)
        x_0 = x_t - sigma * estimated_noise
        diff += (snr(t - np.abs(dt)) - snr(t)) * (torch.norm(point - x_0, dim=1) ** 2)
    sol = -(T / 2) * diff / 2000
    return sol


# The following function train the diffusion model and return the loss process and epochs array
def train_model_uncond():
    # Create the dataset
    points = torch.tensor(Points2Dataset_uncond(number_of_points).points2D)

    # Initialize the Diffusion Model, and optimizer, loss function
    diffusionModel = DiffusionModel_uncond()
    optimizer = optim.Adam(diffusionModel.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss_arr = []
    batches_arr = []

    # Training the Diffusion Model
    for epoch in tqdm(range(number_of_epochs)):
        time = torch.rand((number_of_points, 1))
        noise = torch.randn((number_of_points, 2))
        sigma = np.exp(5 * (time - 1))  # sigma - σ(t)
        x_epoch = points + sigma * noise  # xt=x0+σ(t)·ε ε∼N(0,I)
        noise_predicted = diffusionModel(x_epoch, time).float()   # Forward

        # Compute the loss with MSE loss function
        loss = criterion(noise_predicted, noise.float())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:  # Print loss every 50 epochs
            loss_arr.append(loss.item())
            batches_arr.append(epoch)
    return loss_arr, batches_arr, diffusionModel


# The following function train the diffusion model and return the loss process and epochs array
def train_model_cond():
    # Create the dataset
    p1 = Points2Dataset_cond(200, 0).points2D
    p2 = Points2Dataset_cond(200, 1).points2D
    p3 = Points2Dataset_cond(200, 2).points2D
    p4 = Points2Dataset_cond(200, 3).points2D
    p5 = Points2Dataset_cond(200, 4).points2D
    points = torch.cat((torch.tensor(p1), torch.tensor(p2)), dim=0).float()
    points = torch.cat((points, torch.tensor(p3)), dim=0).float()
    points = torch.cat((points, torch.tensor(p4)), dim=0).float()
    points = torch.cat((points, torch.tensor(p5)), dim=0).float()

    # create appropriate appropriate class
    class_points = np.repeat(np.arange(0, 5), 200)

    diffusionModel = DiffusionModel_cond()
    optimizer = optim.Adam(diffusionModel.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss_arr = []
    batches_arr = []

    # Training the Diffusion Model
    for epoch in tqdm(range(number_of_epochs)):
        time = torch.rand((number_of_points, 1))
        noise = torch.randn((number_of_points, 2))
        # Scheduler - σ(t)
        scheduler = np.exp(5 * (time - 1))
        # xt=x0+σ(t)·ε ε∼N(0,I)
        x_epoch = points + scheduler * noise
        # Forward
        noise_predicted = diffusionModel(x_epoch, time, torch.tensor(class_points).squeeze()).float()

        # Compute the loss with MSE loss function
        loss = criterion(noise_predicted, noise.float())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if epoch % 50 == 0:
            loss_arr.append(loss.item())
            batches_arr.append(epoch)
    return loss_arr, batches_arr, diffusionModel


# Q1:
# For a point of your choice, please present the forward process of it as a trajectory in a 2D space.
# Start from a point outside the square. Color the points according to their time t.
def q1_uncond():
    point = torch.tensor([0, 0])
    points_process = []
    points_process.append(point)
    x_v, y_v = [], []
    ts = [i/1000 for i in range(1000)]
    for t in ts:
        sigma = np.exp(5*(dt-1))
        noise = torch.randn(size=(2,))
        cur_point = point + sigma * noise
        # points_process.append(cur_point)
        x_v.append(cur_point[0])
        y_v.append(cur_point[1])
        point = cur_point
    plt.scatter(x_v, y_v, c=ts, cmap='viridis', s=3)
    plt.plot(x_v, y_v, c='black', linestyle='-',linewidth=0.2)
    plt.colorbar(label='Time (t)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Forward process')
    plt.show()

# Q2:
# Present the loss function over the training batches of the denoiser.
def q2_uncond(batches_arr, losses):
    plt.plot(batches_arr, losses, color='red')
    plt.title('Loss function over the training batches of the denoiser')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.show()


# Q3:
# Present a figure with 9 (3x3 table) different samplings of 1000 points,
# using 9 different seeds.
def q3_uncond():
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            np.random.seed(i+j)
            x = np.random.uniform(-1, 1, 1000)
            y = np.random.uniform(-1, 1, 1000)
            axs[i, j].scatter(x, y, s=2, color='green', alpha=0.5)
            axs[i, j].set(xlabel='X', ylabel='Y')
        plt.grid(True)
    plt.show()


# Q4:
# Show sampling results for different numbers of sampling steps, T.
def q4_uncond():
    Ts = [10, 100, 500, 1000]
    t = 0
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i in range(2):
      for j in range(2):
        ax = axs[i, j]
        ax.set_title(f" T = {Ts[t]}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        dt = 1.0 / Ts[t]
        res, process_z = reverse_process_DDIM_uncond(diffusionModel_uncond, dt, None)
        t += 1
        ax.scatter(res[:, 0], res[:, 1], color='orange', s=5)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    plt.show()


# Slightly modify the given sampler, or create a sampling schedule of your own.
# Plot the σ coefficients over time (sample them using some dt step size),
# in the original sampler, and your modified version.
def q5_uncond():
    t = 1
    ts = []
    orig_samp = []
    small_samp = []
    big_samp = []
    while t >= 0:
        ts.append(t)
        orig_samp.append(np.exp(5 * (t - 1)))
        small_samp.append(np.exp(0.2 * (t-1)))
        big_samp.append(np.exp(10 * (t-1)))
        t -= dt
    plt.plot(ts, orig_samp, label='Original Sampler', c='red')
    plt.plot(ts, small_samp, label=' My Sampler', c='blue')
    plt.plot(ts, big_samp, label=' My Sampler', c='green')
    plt.xlabel('time')
    plt.ylabel('σ coefficients')
    plt.title('σ coefficients over time')
    plt.show()


# Q6:
# insert the same input noise to the reverse sampling process 10 times,
# and plot the outputs.
def q6_uncond():
    tracks = []
    z = torch.tensor([[-0.9, 0.3]], dtype=torch.float64, requires_grad=True)
    for i in range(10):
        x, track_z = reverse_process_DDIM_uncond(diffusionModel_uncond, dt, z)
        if i % 2 == 0 and i != 0:
            tracks.append(track_z)
    plt.figure(figsize=(6, 6))
    for i, track in enumerate(tracks):
        times = [point[1] for point in track]
        plt.scatter(track[:, 0], track[:, 1], c=times, cmap='viridis', label=f'Track {i + 1}')

    plt.colorbar(label='Time (t)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


# Q1:
#  Plot your input coloring the points by their classes.
def q1_cond():
    p1 = Points2Dataset_cond(200, 0)
    p2 = Points2Dataset_cond(200, 1)
    p3 = Points2Dataset_cond(200, 2)
    p4 = Points2Dataset_cond(200, 3)
    p5 = Points2Dataset_cond(200, 4)
    plt.scatter(p1[:, 0], p1[:, 1], color='red')
    plt.scatter(p2[:, 0], p2[:, 1], color='yellow')
    plt.scatter(p3[:, 0], p3[:, 1], color='orange')
    plt.scatter(p4[:, 0], p4[:, 1], color='green')
    plt.scatter(p5[:, 0], p5[:, 1], color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# Q3:
# Sample 1 point from each class. Plot the trajectory of the points, coloring each trajectory with its class’s color.
# Validate the points reach their class region.
def q3_cond():
    # rand 5 points in range (-2,2)
    points = torch.tensor(np.random.uniform(-2, 2, (5, 2)))
    colors = ['red', 'yellow', 'orange', 'green', 'blue']

    dt = 1 / 1000
    random_points = {}
    for c, point in enumerate(points):
        c = torch.tensor(c).int()
        x_t = point.unsqueeze(0)
        random_points[c] = x_t
    # denoise the points and plot the trajectory of each point
    for c, point in random_points.items():
        c = torch.tensor(c).int().unsqueeze(0).unsqueeze(0)
        z, process_z = reverse_process_DDIM_cond(diffusionModel_cond, dt, point, c)
        plt.plot([x[0] for x in process_z], [y[1] for y in process_z], color=colors[c])
        # paint the final point in black
        plt.scatter(z[0], z[1], color='black', s=10)

    plt.xlabel('X')
    plt.ylabel('Y')
    # print the colors square
    print_class_c()
    plt.grid(True)
    plt.show()


# Q4:
# Plot a sampling of at least 1000 points from your trained conditional model.
# Plot the sampled points coloring them by their classes.
def q4_cond():
    dt = 1/1000
    colors_array = ['red', 'yellow', 'orange', 'green', 'blue']
    colors = []
    points, process_z = reverse_process_DDIM_cond(diffusionModel_cond, dt, None, None)
    points = torch.tensor(points)
    points.requires_grad = True
    colors_number = create_colors_array(points)
    for c in colors_number:
        colors.append(colors_array[c])
    plt.scatter(points[:, 0].detach().numpy(), points[:, 1].detach().numpy(), s=5, color=colors, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# Q6:
# Estimate the probability of at least 5 points. Plot the points on top of the data.
# Choose the set of points so some of them are within the input distribution and others are out of it.
# Also, choose a pair of points with the same location but different classes,
# one that matches the input distribution and one that isn’t.
def q6_cond():
    colors_array = ['red', 'yellow', 'orange', 'green', 'blue']
    points = torch.tensor([[3, 0], [-0.3, 0], [0.6, -0.7], [0, 1], [-3, 0]])
    class_array = create_colors_array(points)
    colors = []

    for c in class_array:
        colors.append(colors_array[c])
    for c, point in enumerate(points):
        true_subspace = class_array[c]
        true_p = calc_p(point, torch.tensor([0]).unsqueeze(0), diffusionModel_cond, torch.tensor([true_subspace]))
        wrong_p = []
        for i in range(5):
            if i != true_subspace:
                wrong_p.append(calc_p(point, torch.tensor([0]).unsqueeze(0), diffusionModel_cond, torch.tensor([i])).detach()[0])
        w = torch.tensor(wrong_p).max(dim=0)[0]
        print(f'point [{point[0]}, {point[1]}] true class {true_subspace} color: {colors[c]}\n'
              f' true probability {true_p[0][0].tolist()}, wrong probability {w}')


if __name__ == '__main__':
    # given parameters
    number_of_points = 1000  # suppose to be between 1000-3000 points
    number_of_epochs = 3000
    batch_size = number_of_points  # all samples
    learning_rate = 1e-3
    T = 1000
    dt = 1.0 / T
    power_scheduler = 5

    # train the unconditional model
    loss_arr_uncond, batches_arr_uncond, diffusionModel_uncond = train_model_uncond()
    # z_uncond, process_z_uncond = reverse_process_DDIM_uncond(diffusionModel_uncond, dt, None)
    # q1_uncond()
    # q2_uncond(batches_arr_uncond, loss_arr_uncond)
    # q3_uncond()
    # q4_uncond()
    # q5_uncond()
    # q6_uncond()

    # train the conditional model
    loss_arr_cond, batches_arr_cond, diffusionModel_cond = train_model_cond()

    # q1_cond()
    # q3_cond()
    # q4_cond()
    # q6_cond()



