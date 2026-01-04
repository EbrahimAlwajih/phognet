import torch
import torch.nn as nn

from phognet.utils.phog import calculate_phog_bins

from .phog_layers import PHOGLayer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size_main=3,
        kernel_size_skip=1,
        stride=1,
        padding_main=1,
        padding_skip=0,
    ):
        super().__init__()

        # Main branch
        self.conv1_main = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_main,
            stride=stride,
            padding=padding_main,
        )
        self.bn1_main = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2_main = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size_main, stride=1, padding=padding_main
        )
        self.bn2_main = nn.BatchNorm2d(out_channels)

        # Skip/shortcut connection
        self.shortcut1 = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size_skip,
                    stride=stride,
                    bias=False,
                    padding=padding_skip,
                ),
                nn.BatchNorm2d(out_channels),
            )

        # self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_skip, stride=stride, bias=False),
        # self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Main branch
        identity = x
        out_main = self.conv1_main(identity)
        out_main = self.bn1_main(out_main)
        out_main = self.relu(out_main)

        out_main = self.conv2_main(out_main)
        out_main = self.bn2_main(out_main)

        # Skip/shortcut connection
        out_skip = self.shortcut1(identity)
        # out_skip = self.bn_skip(out_skip)

        # Element-wise addition of main and skip paths
        out = out_main + out_skip

        # Final ReLU activation
        out = self.relu(out)
        return out


class Conv1DModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size_main=3,
        kernel_size_skip=1,
        stride=1,
        stride_skip=1,
        padding_main=1,
        padding_skip=0,
    ):
        super().__init__()

        # Main branch
        self.conv1_main = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_main,
            padding=padding_main,
            stride=stride,
        )
        self.bn1_main = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2_main = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size_main,
            padding=padding_main,
            stride=stride,
        )
        self.bn2_main = nn.BatchNorm1d(out_channels)

        # Skip/shortcut connection

        self.conv_skip = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_skip,
            stride=stride_skip,
            padding=padding_skip,
            bias=False,
        )
        self.bn_skip = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Main branch
        out_main = self.conv1_main(x)
        out_main = self.bn1_main(out_main)
        out_main = self.relu(out_main)

        out_main = self.conv2_main(out_main)
        out_main = self.bn2_main(out_main)

        # Skip/shortcut connection
        out_skip = self.conv_skip(x)
        out_skip = self.bn_skip(out_skip)

        # Element-wise addition of main and skip paths
        out = out_main + out_skip

        # Final ReLU activation
        out = self.relu(out)
        return out


class PHOGProcessingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bins, levels, stride=1, ablation_case=None):
        """
        PHOGProcessingBlock with optional ablation cases to disable PHOG-related components.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bins (int): Number of PHOG bins.
            levels (int): Number of pyramid levels.
            stride (int): Stride for convolution.
            ablation_case (str, optional): Specify ablation case ('remove_phog' or None).
        """
        super().__init__()
        self.in_channels = in_channels
        self.phog_blocks = calculate_phog_bins(bins, levels)
        self.conv2d_module = Conv2DModule(in_channels, out_channels, stride=stride)

        # Initialize PHOG and Conv1D layers based on ablation
        if ablation_case == "remove_phog_block":
            self.phog_layer = None
            self.conv1d_module = None
        else:
            self.phog_layer = PHOGLayer(bins, levels, in_channels=out_channels)
            self.conv1d_module = Conv1DModule(
                self.phog_blocks * out_channels,
                out_channels * calculate_phog_bins(1, levels),
                stride=stride,
            )

    def forward(self, x):
        # Conv2D module output
        conv2d_out = self.conv2d_module(x)

        # If PHOG and Conv1D are enabled
        if self.phog_layer is not None and self.conv1d_module is not None:
            # PHOG Layer output
            phog_out = self.phog_layer(conv2d_out)

            # Reshape PHOG output from 4D [batch_size, 1, height, width] to 3D [batch_size, channels, width]
            batch_size, _, height, width = phog_out.size()
            phog_out_reshaped = phog_out.view(batch_size, height * width, 1)

            # Conv1D module output
            conv1d_out = self.conv1d_module(phog_out_reshaped)
        else:
            phog_out = None
            conv1d_out = None  # Conv1D is bypassed in ablation

        return conv2d_out, conv1d_out


class PHOGNetAblation(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes, bins=20, levels=1, nInputPlane=1, ablation_case=None
    ):
        """
        PHOGNet model with support for ablation studies.

        :param block: Processing block type (e.g., PHOGProcessingBlock).
        :param num_blocks: List specifying the number of blocks in each layer.
        :param num_classes: Number of output classes.
        :param bins: Number of orientation bins in PHOG.
        :param levels: Number of pyramid levels in PHOG.
        :param nInputPlane: Number of input channels.
        :param ablation_case: Specify the ablation case to modify the architecture.
        """
        super().__init__()
        self.bins = bins
        self.levels = levels
        self.phog_blocks = calculate_phog_bins(self.bins, self.levels)
        self.Num_OutputPlane = [16, 32, 64, 128]
        self.in_channels = self.Num_OutputPlane[0]
        self.ablation_case = ablation_case
        self.remove_phog_block = None

        # Stem Layer
        self.conv_stem = ConvBlock(
            nInputPlane, self.Num_OutputPlane[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Apply ablation on PHOG in the stem layer (if specified)
        if self.ablation_case == "remove_phog_stem":
            self.phog_layer = None
            self.conv1d_stem = None
            self.conv1d_stem_size = 0
        else:
            self.phog_layer = PHOGLayer(
                num_bins=self.bins, levels=self.levels, in_channels=nInputPlane
            )
            self.conv1d_stem = Conv1DModule(self.phog_blocks * nInputPlane, self.Num_OutputPlane[1])
            self.conv1d_stem_size = self.Num_OutputPlane[1]

        # PHOG Processing Blocks
        if self.ablation_case == "remove_phog_block":
            self.remove_phog_block = "remove_phog_block"
        self.layer1 = self._make_layer(
            block,
            self.Num_OutputPlane[0],
            num_blocks[0],
            stride=1,
            bins=self.bins,
            levels=self.levels,
            ablation_case=self.remove_phog_block,
        )
        self.layer2 = self._make_layer(
            block,
            self.Num_OutputPlane[1],
            num_blocks[1],
            stride=2,
            bins=self.bins,
            levels=self.levels,
            ablation_case=self.remove_phog_block,
        )
        self.layer3 = self._make_layer(
            block,
            self.Num_OutputPlane[2],
            num_blocks[2],
            stride=2,
            bins=self.bins,
            levels=self.levels,
            ablation_case=self.remove_phog_block,
        )
        self.layer4 = self._make_layer(
            block,
            self.Num_OutputPlane[3],
            num_blocks[3],
            stride=2,
            bins=self.bins,
            levels=self.levels,
            ablation_case=self.remove_phog_block,
        )

        # Global Pooling and Flatten
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Step 1: Multiply each element of Num_OutputPlane by the corresponding element from num_blocks
        # Calculate PHOG size and apply DenseBlock ablation
        if self.ablation_case == "remove_phog_block":
            self.phog_size = 0
        else:
            multiplied_values = [
                self.Num_OutputPlane[i] * num_blocks[i] for i in range(len(self.Num_OutputPlane))
            ]
            sum_multiplied_values = sum(multiplied_values)
            self.phog_size = sum_multiplied_values * calculate_phog_bins(1, self.levels)

        self.dense_block = DenseBlock(
            in_features=self.Num_OutputPlane[3] + self.phog_size + self.conv1d_stem_size,
            num_classes=num_classes,
        )

        # self.dense_block = DenseBlock(in_features=self.Num_OutputPlane[3] + phog_size + self.Num_OutputPlane[1], num_classes=num_classes)

    def _make_layer(
        self, block, out_channels, num_blocks, stride=1, bins=20, levels=1, ablation_case=None
    ):
        layers = []

        # Add the first block with stride
        # print(f"Creating first block: in_channels={self.in_channels}, out_channels={out_channels}, stride={stride}")
        layers.append(
            block(
                bins=bins,
                levels=levels,
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                ablation_case=ablation_case,
            )
        )

        # Update the input channels for the next block
        self.in_channels = out_channels

        # Add remaining blocks with stride=1
        for _ in range(1, num_blocks):
            # print(f"Creating additional block: in_channels={self.in_channels}, out_channels={out_channels}, stride=1")
            layers.append(
                block(
                    bins=bins,
                    levels=levels,
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    ablation_case=ablation_case,
                )
            )

        return CustomSequential(*layers)

    def forward(self, x):
        # Stem Layer
        conv_stem_out = self.conv_stem(x)
        pool_stem_out = self.max_pool(conv_stem_out)
        phog_out_stem = None
        conv1d_stem_out = None

        # Use the PHOG and Conv1D stem block only if both are enabled
        if self.phog_layer is not None and self.conv1d_stem is not None:
            phog_out_stem = self.phog_layer(x).view(x.size(0), -1)  # Compute PHOG
            phog_out_stem = phog_out_stem.unsqueeze(
                2
            )  # Add channel dimension for Conv1D compatibility
            conv1d_stem_out = self.conv1d_stem(phog_out_stem)
        all_phog_outputs = []
        if conv1d_stem_out is not None:
            all_phog_outputs.append(conv1d_stem_out)

        # PHOG Processing Block (PPB)
        # PHOG Processing Block (PPB)
        for layer in self.layer1:
            conv2d_out, phog_out = layer(pool_stem_out)
            if phog_out is not None:
                all_phog_outputs.append(phog_out)
        for layer in self.layer2:
            conv2d_out, phog_out = layer(conv2d_out)
            if phog_out is not None:
                all_phog_outputs.append(phog_out)
        for layer in self.layer3:
            conv2d_out, phog_out = layer(conv2d_out)
            if phog_out is not None:
                all_phog_outputs.append(phog_out)
        for layer in self.layer4:
            conv2d_out, phog_out = layer(conv2d_out)
            if phog_out is not None:
                all_phog_outputs.append(phog_out)

        # conv2d = torch.flatten(conv2d, 1)
        # x_concat = torch.cat((conv2d, phog_concat, phog_input2), dim=1)

        # Concatenate PHOG outputs
        # Concatenate PHOG outputs
        if all_phog_outputs:
            phog_concat = torch.cat([self.flatten(p) for p in all_phog_outputs], dim=1)
        else:
            phog_concat = torch.zeros(x.size(0), 1).to(
                x.device
            )  # Default to a zero tensor if no PHOG output

        # phog_concat = torch.cat([self.flatten(p) for p in all_phog_outputs], dim=1)
        # phog_concat = torch.cat([self.flatten(p) for p in all_phog_outputs if p is not None], dim=1) if all_phog_outputs else None

        # Average Pooling and Flatten
        avg_pooled = self.avg_pool(conv2d_out)
        avg_pooled_flat = self.flatten(avg_pooled)

        # Concatenate outputs
        if phog_concat is not None:
            final_concat = torch.cat([avg_pooled_flat, phog_concat], dim=1)
        else:
            final_concat = avg_pooled_flat

        # Pass through DenseBlock
        output = self.dense_block(final_concat)

        return output


class DenseBlock(nn.Module):
    def __init__(self, in_features, num_classes, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CustomSequential(nn.Sequential):
    def forward(self, input1):
        for module in self:
            input1, input2 = module(input1)
        return input1, input2


class PHOGNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, bins=20, levels=1, nInputPlane=1):
        super().__init__()
        self.bins = bins
        self.levels = levels
        self.phog_blocks = calculate_phog_bins(self.bins, self.levels)
        self.Num_OutputPlane = [16, 32, 64, 128]
        self.in_channels = self.Num_OutputPlane[0]
        # Stem Layer
        self.conv_stem = ConvBlock(
            nInputPlane, self.Num_OutputPlane[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.phog_layer = PHOGLayer(num_bins=self.bins, levels=self.levels, in_channels=nInputPlane)

        self.conv1d_stem = Conv1DModule(self.phog_blocks * nInputPlane, self.Num_OutputPlane[1])

        # PHOG Processing Blocks

        self.layer1 = self._make_layer(
            block,
            self.Num_OutputPlane[0],
            num_blocks[0],
            stride=1,
            bins=self.bins,
            levels=self.levels,
        )
        self.layer2 = self._make_layer(
            block,
            self.Num_OutputPlane[1],
            num_blocks[1],
            stride=2,
            bins=self.bins,
            levels=self.levels,
        )
        self.layer3 = self._make_layer(
            block,
            self.Num_OutputPlane[2],
            num_blocks[2],
            stride=2,
            bins=self.bins,
            levels=self.levels,
        )
        self.layer4 = self._make_layer(
            block,
            self.Num_OutputPlane[3],
            num_blocks[3],
            stride=2,
            bins=self.bins,
            levels=self.levels,
        )

        # Global Pooling and Flatten
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Step 1: Multiply each element of Num_OutputPlane by the corresponding element from num_blocks
        multiplied_values = [
            self.Num_OutputPlane[i] * num_blocks[i] for i in range(len(self.Num_OutputPlane))
        ]
        sum_multiplied_values = sum(multiplied_values)
        phog_size = sum_multiplied_values * calculate_phog_bins(1, self.levels)

        # Dense Block (DB)
        # self.dense_block = DenseBlock(2560, num_classes=num_classes)
        self.dense_block = DenseBlock(
            in_features=self.Num_OutputPlane[3] + phog_size + self.Num_OutputPlane[1],
            num_classes=num_classes,
        )

    def _make_layer(self, block, out_channels, num_blocks, stride=1, bins=20, levels=1):
        layers = []

        # Add the first block with stride
        # print(f"Creating first block: in_channels={self.in_channels}, out_channels={out_channels}, stride={stride}")
        layers.append(
            block(
                bins=bins,
                levels=levels,
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )

        # Update the input channels for the next block
        self.in_channels = out_channels

        # Add remaining blocks with stride=1
        for _ in range(1, num_blocks):
            # print(f"Creating additional block: in_channels={self.in_channels}, out_channels={out_channels}, stride=1")
            layers.append(
                block(
                    bins=bins,
                    levels=levels,
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                )
            )

        return CustomSequential(*layers)

    def forward(self, x):
        # Stem Layer
        conv_stem_out = self.conv_stem(x)
        pool_stem_out = self.max_pool(conv_stem_out)
        phog_out_stem = self.phog_layer(x).view(x.size(0), -1)
        phog_out_stem = phog_out_stem.unsqueeze(2)
        conv1d_stem_out = self.conv1d_stem(phog_out_stem)

        all_phog_outputs = [conv1d_stem_out]

        # PHOG Processing Block (PPB)
        # all_phog_outputs = []
        for layer in self.layer1:
            conv2d_out, phog_out = layer(pool_stem_out)
            all_phog_outputs.extend([phog_out])
        for layer in self.layer2:
            conv2d_out, phog_out = layer(conv2d_out)
            all_phog_outputs.extend([phog_out])
        for layer in self.layer3:
            conv2d_out, phog_out = layer(conv2d_out)
            all_phog_outputs.extend([phog_out])
        for layer in self.layer4:
            conv2d_out, phog_out = layer(conv2d_out)
            all_phog_outputs.extend([phog_out])

        # conv2d = torch.flatten(conv2d, 1)
        # x_concat = torch.cat((conv2d, phog_concat, phog_input2), dim=1)

        # Concatenate PHOG outputs
        phog_concat = torch.cat([self.flatten(p) for p in all_phog_outputs], dim=1)

        # Average Pooling and Flatten
        avg_pooled = self.avg_pool(conv2d_out)
        avg_pooled_flat = self.flatten(avg_pooled)

        # Concatenate avg_pooled with PHOG outputs
        final_concat = torch.cat([avg_pooled_flat, phog_concat], dim=1)

        # Pass through DenseBlock
        output = self.dense_block(final_concat)

        return output
