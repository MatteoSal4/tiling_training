import torch
import torch.nn as nn
import torch.nn.functional as F
from network_elements import MultiScaleElaboration, Reduction, Expansion

class unet(nn.Module):
    def __init__(self, input_shape=(128,16,128)):
        super().__init__()
        self.D, self.H, self.W = input_shape


        # Encoder

        self.enc_multi_elab_1 = MultiScaleElaboration(name = "enc_multi_elab_enc_1")
        self.enc_multi_elab_2 = MultiScaleElaboration(name = "enc_multi_elab_enc_2")

        self.reduction_1 = Reduction(name = "reduction_1")

        self.enc_multi_elab_3 = MultiScaleElaboration(name = "enc_multi_elab_enc_3")
        self.enc_multi_elab_4 = MultiScaleElaboration(name = "enc_multi_elab_enc_4")

        self.reduction_2 = Reduction(name = "reduction_2")

        self.enc_multi_elab_5 = MultiScaleElaboration(name = "enc_multi_elab_enc_5")
        self.enc_multi_elab_6 = MultiScaleElaboration(name = "enc_multi_elab_enc_6")

        self.reduction_3 = Reduction(name = "ct_reduction_3")

        self.bottleneck_multi_elab_1 = MultiScaleElaboration(name = "bottleneck_multi_elab_1")
        self.bottleneck_multi_elab_2 = MultiScaleElaboration(name = "bottleneck_multi_elab_2")

        # Decoder

        self.expansion_1 = Expansion(name = "expansion_1")

        self.dec_multi_elab_1 = MultiScaleElaboration(name = "dec_multi_elab_1")
        self.dec_multi_elab_2 = MultiScaleElaboration(name = "dec_multi_elab_2")

        self.expansion_2 = Expansion(name = "expansion_2")

        self.dec_multi_elab_3 = MultiScaleElaboration(name = "dec_multi_elab_3")
        self.dec_multi_elab_4 = MultiScaleElaboration(name = "dec_multi_elab_4")

        self.expansion_3 = Expansion(name = "expansion_3")

        self.dec_multi_elab_5 = MultiScaleElaboration(name = "dec_multi_elab_5")
        self.dec_multi_elab_6 = MultiScaleElaboration(name = "dec_multi_elab_6")

        self.conv = nn.LazyConv3d(out_channels=2, kernel_size=1)        # Two channel then one channel output
        self.final_conv = nn.LazyConv3d(out_channels=1, kernel_size=1)
    
    def forward(self, x):

        ################### Encoder ###################
        x1 = self.enc_multi_elab_1(x)
        x1 = self.enc_multi_elab_2(x1)

        red_1 = self.reduction_1(x1)

        x2 = self.enc_multi_elab_3(red_1)
        x2 = self.enc_multi_elab_4(x2)

        red_2 = self.reduction_2(x2)

        x3 = self.enc_multi_elab_5(red_2)
        x3 = self.enc_multi_elab_6(x3)

        red3 = self.reduction_3(x3)

        bottleneck = self.bottleneck_multi_elab_1(red3)
        bottleneck = self.bottleneck_multi_elab_2(bottleneck)

        ################### Decoder ###################
        exp1 = self.expansion_1(bottleneck)
    
        concat1 = torch.cat((exp1, x3), dim=1)

        dec_multi_elab_1 = self.dec_multi_elab_1(concat1)
        dec_multi_elab_2 = self.dec_multi_elab_2(dec_multi_elab_1)

        exp2 = self.expansion_2(dec_multi_elab_2)

        concat2 = torch.cat((exp2, x2), dim=1)

        dec_multi_elab_3 = self.dec_multi_elab_3(concat2)
        dec_multi_elab_4 = self.dec_multi_elab_4(dec_multi_elab_3)

        exp3 = self.expansion_3(dec_multi_elab_4)

        concat3 = torch.cat((exp3, x1), dim=1)

        dec_multi_elab_5 = self.dec_multi_elab_5(concat3)
        dec_multi_elab_6 = self.dec_multi_elab_6(dec_multi_elab_5)

        out = self.conv(dec_multi_elab_6)
        out = self.final_conv(out)

        return out