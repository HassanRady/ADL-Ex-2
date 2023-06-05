import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    t = torch.linspace(0, timesteps, timesteps + 1)
    alpha = torch.cos((t/timesteps + s) / (1 + s) * (torch.pi/2)) ** 2
    alpha = alpha / alpha[0]
    beta = 1 - (alpha[1:] / alpha[:-1])   # [1:] because at t=1 [:-1] will be 0
    return torch.clip(beta, 0.0001, 0.9999)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """

    # inspired from this paper https://arxiv.org/abs/2301.10972
    # v_start = torch.sigmoid(beta_start)
    # v_end = torch.sigmoid(beta_end)
    # beta = torch.sigmoid((timesteps * (beta_end - beta_start) + beta_start))
    # beta = (v_end - beta) / (v_end - v_start)
    # return torch.clip(beta, 0.0001, 0.9999)
    t = torch.linspace(0, timesteps, timesteps + 1)
    s_limit = 6  # from the blog
    beta = beta_start + \
        torch.sigmoid(-s_limit + 2*t*s_limit/timesteps) * \
        (beta_end - beta_start)
    return beta


class Diffusion:

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        # define alphas
        alpha = 1 - self.betas
        alphas = torch.cumprod(alpha, axis=0)
        self.one_over_alpha_sqrt = torch.sqrt(1/alpha)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.alphas_sqrt = torch.sqrt(alphas)
        self.one_minus_alphas_sqrt = torch.sqrt(1 - alphas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        prev_alphas = torch.cat([torch.tensor([1.0]), alphas[:-1]])
        self.sigma = self.betas * (1 - prev_alphas)/(1 - alphas)

    def get_ts(self, alphas, t, shape):
        batch_size = t.shape[0]
        return alphas.gather(
                -1, t.cpu()).reshape(batch_size, *((1,) * (len(shape) - 1))).to(self.device)

    @torch.no_grad()
    def p_sample(self, model, x, t, c, t_index):

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        ts_betas = self.get_ts(self.betas, t, x.shape)
        ts_one_minus_alphas_sqrt = self.get_ts(self.one_minus_alphas_sqrt, t, x.shape)
        ts_one_over_alphas_sqrt = self.get_ts(self.one_over_alpha_sqrt, t, x.shape)
        mean = ts_one_over_alphas_sqrt * (x - ts_betas * model(x, t, c) / ts_one_minus_alphas_sqrt)

        if t_index == 0:
            return mean
        else:
            z = torch.randn_like(x)
            ts_sigma = self.get_ts(self.sigma, t, x.shape)
            return mean + z * ts_sigma

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, c, image_size, batch_size=16, channels=3):
        xs = []
        x = torch.randn((batch_size, channels, image_size, image_size), device=self.device)
        for i in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps):
            t = torch.full((batch_size), i, device=self.device)
            x = self.p_sample(model, x, t, c, i)
            xs.append(x.detach().numpy())
        return xs

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        if noise is None:
            noise = torch.rand_like(x_zero).to(self.device)

        x0_coefficient = self.get_ts(self.alphas_sqrt, t, x_zero.shape)
        noise_coefficient = self.get_ts(self.one_minus_alphas_sqrt, t, x_zero.shape)

        return x0_coefficient * x_zero + noise_coefficient * noise

    def p_losses(self, denoise_model, x_zero, t, c, noise=None, loss_type="l1"):

        if noise is None:
            noise = torch.rand_like(x_zero).to(self.device)

        x = self.q_sample(x_zero, t, noise)
        pred = denoise_model(x, t, c)

        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(noise, pred)
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(noise, pred)
        else:
            raise NotImplementedError()

        return loss
