from __future__ import absolute_import, division
import torch

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
             _reset(nn)

class Encoder(torch.nn.Module):
    def __init__(self,in_channel, hidden_layers, latent_dim, dropRate=0.1, dropIn=0, bn=0.6, activation=None, bias=True, dtype=torch.double):
        super(Encoder, self).__init__()

        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias, dtype=dtype))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn, dtype=dtype))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias, dtype=dtype))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn, dtype=dtype))

        self.linear_latent_mu = torch.nn.Linear(hidden_layers[-1], latent_dim, bias=False, dtype=dtype)
        if activation is not None:
            self.activation = activation
        self.dropout = torch.nn.Dropout(dropRate)
        self.drop_in = dropIn
        if dropIn > 0:
            self.drop_input = torch.nn.Dropout(dropIn)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0.

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.drop_in > 0:
            x = self.drop_input(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        z_latent = self.linear_latent_mu(x)
        return z_latent

    def L2Regularization(self, L2):
        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss += L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias:
                biasLoss += L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return L2Loss


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_layers, out_dim, dropRate=0.1, dropIn=0, bn=0.6, activation=None, bias=True, dtype=torch.double):
        super(Decoder, self).__init__()

        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(latent_dim, hidden_layers[0], bias=bias, dtype=dtype))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn, dtype=dtype))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias, dtype=dtype))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn, dtype=dtype))
        self.output_linear = torch.nn.Linear(hidden_layers[-1], out_dim, bias=False, dtype=dtype)

        if activation is not None:
            self.activation = activation
        self.dropout = torch.nn.Dropout(dropRate)
        self.dropIn = dropIn
        if dropIn > 0:
            self.drop_input = torch.nn.Dropout(dropIn)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.dropIn > 0:
            x = self.drop_input(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        output = self.output_linear(x)
        return output

    def L2Regularization(self, L2):
        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss += L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias:
                biasLoss += L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return L2Loss


class VAE(torch.nn.Module):
    def __init__(self, enc, dec, device):
        super(VAE, self).__init__()

        self.encoder = enc
        self.decoder = dec
        self.device = device

    def forward(self, x):
        z_latent = self.encoder(x)
        predicted = self.decoder(z_latent)

        return z_latent, predicted

    def encode(self, x):
        z_latent = self.encoder(x)
        return z_latent

    def decode(self, x):
        decoded_output = self.decoder(x)
        return decoded_output

    def L2Regularization(self, L2):
        encoderL2 = self.encoder.L2Regularization(L2)
        decoderL2 = self.decoder.L2Regularization(L2)

        L2Loss = encoderL2 + decoderL2
        return L2Loss


class SimpleEncoder(torch.nn.Module):
    def __init__(self, in_channel, hidden_layers, latent_dim, dropRate=0.1, dropIn=0, bn=0.6, activation=None, normalizeOutput=False, bias=True, dtype=torch.double):
        super(SimpleEncoder, self).__init__()

        self.bias = bias
        self.normalizeOutput = normalizeOutput
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias, dtype=dtype))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn, dtype=dtype))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias, dtype=dtype))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn, dtype=dtype))

        self.linear_latent = torch.nn.Linear(hidden_layers[-1], latent_dim, bias=False, dtype=dtype)
        if activation is not None:
            self.activation = activation
        self.dropout = torch.nn.Dropout(dropRate)
        self.drop_in = dropIn
        if dropIn > 0:
            self.drop_input = torch.nn.Dropout(dropIn)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.drop_in > 0:
            x = self.drop_input(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        z_latent = self.linear_latent(x)
        if self.normalizeOutput:
            z_latent = torch.nn.functional.normalize(z_latent)
        return z_latent

    def L2Regularization(self, L2):
        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss += L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias:
                biasLoss += L2 * torch.sum((self.linear_layers[i].bias)**2)
        weightLoss += L2 * torch.sum((self.linear_latent.weight)**2)
        L2Loss = biasLoss + weightLoss
        return L2Loss


class Classifier(torch.nn.Module):
    def __init__(self, in_channel, hidden_layers, num_classes, drop_in=0.5, drop=0.2, bn=0.6, bias=True, dtype=torch.double):
        super(Classifier, self).__init__()
        self.drop_in = drop_in
        self.num_hidden_layers = len(hidden_layers)
        self.bias = bias
        self.num_classes = num_classes
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias, dtype=dtype))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn, dtype=dtype))
        self.dropouts.append(torch.nn.Dropout(drop))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias, dtype=dtype))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn, dtype=dtype))
            self.dropouts.append(torch.nn.Dropout(drop))

        self.linear_output = torch.nn.Linear(hidden_layers[-1], num_classes, bias=False, dtype=dtype)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.drop_in > 0:
            x = torch.nn.functional.dropout(x, p=self.drop_in)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)
        output = self.linear_output(x)
        return output
    
    def L2Regularization(self, L2):
        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss += L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias:
                biasLoss += L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        if isinstance(self.linear_output, torch.nn.Linear):
            L2Loss += L2 * torch.sum((self.linear_output.weight)**2)
            if self.linear_output.bias is not None:
                L2Loss += L2 * torch.sum((self.linear_output.bias)**2)
        return L2Loss
    
class SimpleANN(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, momentum=0.1, drop_rate=0.2, 
                 activation=torch.nn.LeakyReLU(0.1), 
                 output_activation =None,
                 bias=True, dtype=torch.double):
        super(SimpleANN, self).__init__()
        
        # Feed-forward layers
        self.layers = torch.nn.ModuleList()
        prev_dim = input_dim
        
        # Hidden layers
        for layer_dim in hidden_layers:
            self.layers.append(torch.nn.Linear(prev_dim, layer_dim, bias=bias, dtype=dtype))
            self.layers.append(torch.nn.BatchNorm1d(layer_dim, momentum=momentum, dtype=dtype))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(drop_rate))
            prev_dim = layer_dim
        
        # Output layer
        self.output_layer = torch.nn.Linear(prev_dim, output_dim, bias=bias, dtype=dtype)
        self.output_activation = output_activation
    
    def forward(self, x):
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        output = self.output_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output
    
    def L2Regularization(self, L2):
        weightLoss = 0.
        biasLoss = 0.
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                weightLoss += L2 * torch.sum((layer.weight)**2)
                if layer.bias is not None:
                    biasLoss += L2 * torch.sum((layer.bias)**2)
        L2Loss = biasLoss + weightLoss
        if isinstance(self.output_layer, torch.nn.Linear):
            L2Loss += L2 * torch.sum((self.output_layer.weight)**2)
            if self.output_layer.bias is not None:
                L2Loss += L2 * torch.sum((self.output_layer.bias)**2)
        return L2Loss


class VarDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_layers, out_dim,dropRate=0.1, dropIn=0, bn=0.6, activation=None, bias=True,loss='nb', dtype=torch.double):

        super(VarDecoder, self).__init__()

        if loss not in ['nb','gauss']:
            raise Exception("The only allowed arguments for `loss` are `nb` and `gauss`")
        self.loss = loss
        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.linear_layers = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(latent_dim, hidden_layers[0], bias=bias, dtype=dtype))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn, dtype=dtype))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias, dtype=dtype))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn, dtype=dtype))
        
        self.out_var = torch.nn.Linear(hidden_layers[-1],
                                       out_dim,
                                       bias=False, 
                                       dtype=dtype)
        self.out_mu = torch.nn.Linear(hidden_layers[-1],
                                       out_dim,
                                       bias=False, 
                                       dtype=dtype)

        if activation is not None:
            self.activation = activation
        self.dropout = torch.nn.Dropout(dropRate)
        self.dropIn = dropIn
        if dropIn > 0:
            self.drop_input = torch.nn.Dropout(dropIn)
        self.relu = torch.nn.LeakyReLU()

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.dropIn > 0:
            x = self.drop_input(x)
        if self.dropIn > 0:
            x = self.drop_input(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        if self.loss == 'gauss':
            # convert variance estimates to a positive value in [1e-3, \infty)
            gene_means = torch.nn.functional.softplus(self.relu(self.out_mu(x))).add(1e-3)
            gene_vars = torch.nn.functional.softplus(self.relu(self.out_var(x))).add(1e-3)
        if self.loss == 'nb':
            gene_means = torch.nn.functional.softplus(self.relu(self.out_mu(x))).add(1e-3)
            gene_vars = torch.nn.functional.softplus(self.relu(self.out_var(x))).add(1e-3)        
        return gene_means,gene_vars

    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias==True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return(L2Loss)
 
class SpeciesCovariate(torch.nn.Module):
    def __init__(self,latent_dim1, latent_dim2,dropRate=0.1,dtype=torch.double):

        super(SpeciesCovariate, self).__init__()
        self.Vspecies = torch.nn.Linear(latent_dim1, latent_dim2, bias=False,dtype=dtype)
        self.dropOut = torch.nn.Dropout(dropRate)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, zbasal, zspecies):

        # zspecies = self.MPL(zspecies)
        z_cov = zbasal + self.dropOut(self.Vspecies(zspecies))

        return z_cov

    def Regularization(self, L = 1e-4):

        # Regularize L2 and also regularize not to be zero
        weightLoss = L * torch.sum((torch.square(self.Vspecies.weight)))

        return(weightLoss)
    
class LocalDiscriminator(torch.nn.Module):
    r"""Implemented from https://github.com/BioSysLab/deepSNEM"""
    def __init__(self, input_dim, dim, dtype=torch.double):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim,dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, input_dim,dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, dim,dtype=dtype),
            torch.nn.ReLU()
        )
        self.linear_shortcut = torch.nn.Linear(input_dim, dim,dtype=dtype)
    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class PriorDiscriminator(torch.nn.Module):
    r"""Implemented from https://github.com/BioSysLab/deepSNEM"""
    def __init__(self, input_dim,dtype=torch.double):
        super().__init__()
        self.l0 = torch.nn.Linear(input_dim, input_dim,dtype=dtype)
        self.l1 = torch.nn.Linear(input_dim, input_dim,dtype=dtype)
        self.l2 = torch.nn.Linear(input_dim, 1,dtype=dtype)
    def forward(self, x):
        h = torch.nn.functional.relu(self.l0(x))
        h = torch.nn.functional.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))
    
class Flow(torch.nn.Module):
    r"""Adapted from https://github.com/facebookresearch/flow_matching"""
    def __init__(self, dim: int, h: int,dtype=torch.double):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, h, dtype=dtype), torch.nn.ELU(),
            torch.nn.Linear(h, h, dtype=dtype), torch.nn.ELU(),
            torch.nn.Linear(h, h, dtype=dtype), torch.nn.ELU(),
            torch.nn.Linear(h, dim, dtype=dtype))
        # self.dtype = dtype
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((t, x_t), -1)) 
    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end - t_start) / 2,t_start + (t_end - t_start) / 2)
    

# --- ConditionalFlow Class Definition ---
class ConditionalFlow(torch.nn.Module):
    """
    This Flow class is adapted for conditional generation.
    It takes a condition x_1 as input to guide the flow from a noise vector z_0
    to a target vector z_2. It also includes an ODE solver step for inference.
    """
    def __init__(self, dim_m1: int, dim_m2: int, h: int, dtype=torch.double):
        super().__init__()
        self.dtype = dtype
        
        # The input to the network is the concatenation of:
        # - The state in the target space (dim_m2)
        # - The time t (1)
        # - The condition from the source space (dim_m1)
        input_dim = dim_m2 + 1 + dim_m1
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, h, dtype=self.dtype), torch.nn.ELU(),
            torch.nn.Linear(h, h, dtype=self.dtype), torch.nn.ELU(),
            torch.nn.Linear(h, h, dtype=self.dtype), torch.nn.ELU(),
            torch.nn.Linear(h, dim_m2, dtype=self.dtype)  # The output is a velocity in the target space (dim_m2)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """ The forward pass now includes the condition x_1. """
        if t.ndim == 1:
            t = t.unsqueeze(1)
        # Ensure all inputs are on the same device and of the same dtype
        t = t.to(x_t.device).type(self.dtype)
        x_1 = x_1.to(x_t.device).type(self.dtype)
        
        net_input = torch.cat((x_t, t, x_1), -1)
        return self.net(net_input)

    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, x_1_cond: torch.Tensor) -> torch.Tensor:
        """ 
        ODE solver step (midpoint method) for inference.
        This function now requires the condition `x_1_cond` to generate the velocity.
        """
        h = t_end - t_start
        # Ensure t_start is correctly shaped for the model forward pass
        t_start_exp = t_start.view(1, 1).expand(x_t.shape[0], 1)

        # Midpoint method requires two model evaluations
        v_start = self.forward(x_t, t_start_exp, x_1_cond)
        
        x_mid = x_t + v_start * h / 2.0
        t_mid = t_start + h / 2.0
        t_mid_exp = t_mid.view(1, 1).expand(x_t.shape[0], 1)

        v_mid = self.forward(x_mid, t_mid_exp, x_1_cond)
        
        return x_t + v_mid * h
    
class ElementWiseLinear(torch.nn.Module):
    def __init__(self,dim,leak = 0.1,bias=True,drop = 0.2):

        super(ElementWiseLinear,self).__init__()

        weight = 1e-1*torch.randn(dim)
        self.weight = torch.nn.Parameter(weight)
        if bias==True:
            bias = 1e-3*torch.randn(dim)
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias=None
        
        self.drop = torch.nn.Dropout(drop)
        self.activation = torch.nn.LeakyReLU(leak)

    def forward(self,x):

        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        x = self.drop(x)

        return x
