import torch


class CostGPTrajectory:
    def __init__(self, n_support_points, dt, sigma_gp=None, **kwargs):
        self.n_support_points = n_support_points
        self.n_dof = 7
        self.dim = 14
        self.dt = dt
        self.sigma_gp = sigma_gp
        self.tensor_args = {"dtype": torch.float32, "device": "cuda"}
        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def __call__(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1).squeeze()
        costs = gp_costs
        return costs


class GPFactor:
    def __init__(
        self,
        dim: int,
        sigma,
        d_t,
        num_factors,
        tensor_args,
        Q_c_inv=None,
    ):
        self.dim = dim
        self.d_t = d_t
        self.tensor_args = tensor_args
        self.state_dim = self.dim * 2  # position and velocity
        self.num_factors = num_factors
        self.phi = self.calc_phi()
        if Q_c_inv is None:
            Q_c_inv = torch.eye(dim, **tensor_args) / sigma**2
        self.Q_c_inv = torch.zeros(num_factors, dim, dim, **tensor_args) + Q_c_inv
        self.Q_inv = self.calc_Q_inv()  # shape: [num_factors, state_dim, state_dim]

        ## Pre-compute constant Jacobians
        self.H1 = self.phi.unsqueeze(0).repeat(self.num_factors, 1, 1)
        self.H2 = -1.0 * torch.eye(self.state_dim, **self.tensor_args).unsqueeze(
            0
        ).repeat(self.num_factors, 1, 1)

    def calc_phi(self):
        eye = torch.eye(self.dim, **self.tensor_args)
        Z = torch.zeros(self.dim, self.dim, **self.tensor_args)
        phi_u = torch.cat((eye, self.d_t * eye), dim=1)
        phi_l = torch.cat((Z, eye), dim=1)
        phi = torch.cat((phi_u, phi_l), dim=0)
        return phi

    def calc_Q_inv(self):
        m1 = 12.0 * (self.d_t**-3.0) * self.Q_c_inv
        m2 = -6.0 * (self.d_t**-2.0) * self.Q_c_inv
        m3 = 4.0 * (self.d_t**-1.0) * self.Q_c_inv

        Q_inv_u = torch.cat((m1, m2), dim=-1)
        Q_inv_l = torch.cat((m2, m3), dim=-1)
        Q_inv = torch.cat((Q_inv_u, Q_inv_l), dim=-2)
        return Q_inv

    def get_error(self, x_traj, calc_jacobian=True):
        horizon = x_traj.shape[1]
        idx1 = torch.arange(0, horizon - 1, device=self.tensor_args["device"])
        idx2 = torch.arange(1, horizon, device=self.tensor_args["device"])
        state_1 = torch.index_select(x_traj, 1, idx1).unsqueeze(-1)
        state_2 = torch.index_select(x_traj, 1, idx2).unsqueeze(-1)
        error = state_2 - self.phi @ state_1

        if calc_jacobian:
            H1 = self.H1
            H2 = self.H2
            # H1 = self.H1.unsqueeze(0).repeat(batch, 1, 1, 1)
            # H2 = self.H2.unsqueeze(0).repeat(batch, 1, 1, 1)
            return error, H1, H2
        else:
            return error
