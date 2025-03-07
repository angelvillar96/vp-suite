import numpy as np
import torch
import random

from tqdm import tqdm

from vp_suite.base.base_model import VideoPredictionModel
from vp_suite.model_blocks.enc import DCGANEncoder, DCGANDecoder
from vp_suite.model_blocks.phydnet import K2M, DecoderSplit, EncoderSplit, PhyCell, SingleStepConvLSTM


class PhyDNet(VideoPredictionModel):
    r"""

    """

    # model-specific constants
    NAME = "PhyDNet"
    PAPER_REFERENCE = "https://arxiv.org/abs/2003.01460"  #: The paper where this model was introduced first.
    CODE_REFERENCE = "https://github.com/vincent-leguen/PhyDNet"  #: The code location of the reference implementation.
    MATCHES_REFERENCE: str = "No"  #: A comment indicating whether the implementation in this package matches the reference.
    CAN_HANDLE_ACTIONS = True

    # model hyperparameters
    phycell_n_layers = 1
    phycell_channels = 49
    phycell_kernel_size = (7, 7)
    convlstm_n_layers = 3
    convlstm_hidden_dims = [128, 128, 64]
    convlstm_kernel_size = (3, 3)

    # model training hyperparams
    moment_loss_scale = 1.0
    teacher_forcing_decay = 0.003

    def __init__(self, device, **model_kwargs):
        r"""

        Args:
            device ():
            **model_kwargs ():
        """
        super(PhyDNet, self).__init__(device, **model_kwargs)

        self.encoder_E = DCGANEncoder(nc=self.img_c).to(self.device)
        self.encoder_Ep = EncoderSplit().to(self.device)
        self.encoder_Er = EncoderSplit().to(self.device)

        zeros = torch.zeros((1, *self.img_shape), device=device)
        encoded_zeros = self.encoder_E(zeros)
        self.shape_Ep = self.encoder_Ep(encoded_zeros).shape[1:]
        self.shape_Er = self.encoder_Er(encoded_zeros).shape[1:]

        self.decoder_Dp = DecoderSplit().to(self.device)
        self.decoder_Dr = DecoderSplit().to(self.device)
        self.decoder_D = DCGANDecoder(out_size=self.img_shape[1:], nc=self.img_c).to(self.device)

        phycell_hidden_dims = [self.phycell_channels] * self.phycell_n_layers
        self.phycell = PhyCell(input_size=self.shape_Ep[1:], input_dim=self.shape_Ep[0],
                               hidden_dims=phycell_hidden_dims, n_layers=self.phycell_n_layers,
                               kernel_size=self.phycell_kernel_size, action_conditional=self.action_conditional,
                               action_size=self.action_size, device=device).to(self.device)

        self.convcell = SingleStepConvLSTM(input_size=self.shape_Er[1:], input_dim=self.shape_Ep[0],
                                           hidden_dims=self.convlstm_hidden_dims, n_layers=self.convlstm_n_layers,
                                           kernel_size=self.convlstm_kernel_size,
                                           action_conditional=self.action_conditional,
                                           action_size=self.action_size, device=device).to(self.device)

        self.constraints = torch.zeros((self.phycell_channels, *self.phycell_kernel_size), device=self.device)
        ind = 0
        for i in range(0, self.phycell_kernel_size[0]):
            for j in range(0, self.phycell_kernel_size[1]):
                self.constraints[ind, i, j] = 1
                ind += 1

    def encoder_fwd(self, frame, action, first_timestep=False, decoding=False):
        r"""

        Args:
            frame ():
            action ():
            first_timestep ():
            decoding ():

        Returns:

        """
        frame = self.encoder_E(frame)  # general encoder 64x64x1 -> 32x32x32
        input_phys = None if decoding else self.encoder_Ep(frame)
        input_conv = self.encoder_Er(frame)

        hidden1, output1 = self.phycell(input_phys, action, first_timestep)
        hidden2, output2 = self.convcell(input_conv, action, first_timestep)

        decoded_phys = self.decoder_Dp(output1[-1])
        decoded_conv = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(self.decoder_D(decoded_phys))  # partial reconstructions for visualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_conv))

        concat = decoded_phys + decoded_conv
        output_image = torch.sigmoid(self.decoder_D(concat))
        return out_phys, hidden1, output_image, out_phys, out_conv

    def pred_1(self, x, **kwargs):
        r"""

        Args:
            x ():
            **kwargs ():

        Returns:

        """
        return self(x, pred_frames=1, **kwargs)[0].squeeze(dim=1)

    def forward(self, x, pred_frames=1, **kwargs):
        r"""

        Note:
            For this model, forward() is used for inference only (no training).

        Args:
            x ():
            pred_frames ():
            **kwargs ():

        Returns:

        """

        # in training mode (default: False), returned sequence starts with 2nd context frame,
        # and the moment regularization loss is calculated.
        train = kwargs.get("train", False)
        teacher_forcing = kwargs.get("teacher_forcing", False) and train
        context_frames = x.shape[1] - pred_frames if train else x.shape[1]
        empty_actions = torch.zeros(x.shape[0], context_frames + pred_frames - 1, device=self.device)
        actions = kwargs.get("actions", empty_actions)
        if self.action_conditional:
            if actions.equal(empty_actions) or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")

        out_frames = []
        ac_index = 0
        for ei in range(context_frames - 1):
            encoder_output, encoder_hidden, output_image, _, _ = \
                self.encoder_fwd(x[:, ei], actions[:, ac_index], (ac_index == 0))
            if train:
                out_frames.append(output_image)
            ac_index += 1

        decoder_input = x[:, context_frames - 1]  # first decoder input = last context frame

        for di in range(pred_frames):
            decoder_output, decoder_hidden, output_image, _, _ = \
                self.encoder_fwd(decoder_input, actions[:, ac_index], (ac_index == 0))
            out_frames.append(output_image)
            decoder_input = x[:, context_frames + di] if teacher_forcing else output_image
            ac_index += 1
        out_frames = torch.stack(out_frames, dim=1)

        # Moment regularization loss during training
        if train:
            k2m = K2M(self.phycell_kernel_size).to(self.device)
            moment_loss = 0
            for b in range(0, self.phycell.cell_list[0].input_dim):
                filters = self.phycell.cell_list[0].F.conv1.weight[:, b]
                moment = k2m(filters.double()).float()
                moment_loss += torch.mean(self.moment_loss_scale * (moment - self.constraints) ** 2)
            model_losses = {"moment regularization loss": moment_loss}
        else:
            model_losses = None

        return out_frames, model_losses

    def train_iter(self, config, data_loader, optimizer, loss_provider, epoch):
        r"""

        Args:
            config ():
            data_loader ():
            optimizer ():
            loss_provider ():
            epoch ():

        Returns:

        """

        teacher_forcing_ratio = np.maximum(0, 1 - epoch * self.teacher_forcing_decay)
        loop = tqdm(data_loader)
        for batch_idx, data in enumerate(loop):

            # fwd
            input_frames, _, actions = self.unpack_data(data, config, complete=True)
            teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            predictions, model_losses = self(input_frames, pred_frames=config["pred_frames"],
                                             actions=actions, train=True, teacher_forcing=teacher_forcing)

            # loss
            targets = input_frames[:, 1:]  # image-wise loss are taken from 2nd context frame onwards
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            # bwd
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # bookkeeping
            loop.set_postfix(loss=total_loss.item())
