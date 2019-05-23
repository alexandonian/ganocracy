from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


def proj(x, y):
    """Projection of x onto y."""
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    """Orthogonalize x wrt list of vectors ys."""
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    """Apply num_itrs steps of the power method to estimate top N singular values."""
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration.
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors.
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors.
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


class SN(object):
    """Spectral normalization base class.

    Layers should inherit from this class.
    """

    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u{}'.format(i), torch.randn(1, num_outputs))
            self.register_buffer('sv{}'.format(i), torch.ones(1))

    @property
    def u(self):
        """Singular vectors (u side)."""
        return [getattr(self, 'u{}'.format(i)) for i in range(self.num_svs)]

    @property
    def sv(self):
        """Singular values.
        Note: that these buffers are just for logging and are not used in training.
        """
        return [getattr(self, 'sv{}'.format(i)) for i in range(self.num_svs)]

    def W_(self):
        """Compute the spectrally-normalized weight."""

        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    """Conv2d layer with spectral normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    """Linear layer with spectral normalization."""

    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class SNEmbedding(nn.Embedding, SN):
    """Embedding layer with spectral norm.

    We use num_embeddings as the dim instead of embedding_dim here
    for convenience sake
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                              max_norm, norm_type, scale_grad_by_freq,
                              sparse, _weight)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1,
                 linear_func=SNLinear):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
        self.gamma_embed = linear_func(num_classes, num_features, bias=False)
        self.beta_embed = linear_func(num_classes, num_features, bias=False)

    def forward(self, x, y):
        # First, compute standard batchnorm stats.
        out = self.bn(x)

        # Learn class specific scale and shift.
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


def l2normalize(v, eps=1e-4):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """An alternate implementation of spectral normalization.

    To apply spectral norm, this class should wrap the layer instance.
    Prefer the other implementation unless there is a specific reason
    not to.
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        _w = w.view(height, -1)
        for _ in range(self.power_iterations):
            v = l2normalize(torch.matmul(_w.t(), u))
            u = l2normalize(torch.matmul(_w, v))

        sigma = u.dot((_w).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
