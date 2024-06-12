
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class FreeMono3DHead(nn.Module):
    def __init__(self, 
            num_classes=1,
            in_channels=128,
            feat_channels=256,
            stacked_convs=4,
            strides=(4, 8, 16, 32, 64),
            dcn_on_last_conv=False,
            conv_bias='auto',
            background_label=None,
            use_direction_classifier=True,
            diff_rad_by_sin=True,
            dir_offset=0,
            dir_limit_offset=0,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_dir=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_attr=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            bbox_code_size=9,  # For nuscenes
            pred_attrs=False,
            num_attrs=1,  # For nuscenes
            pred_velo=False,
            pred_bbox2d=False,
            group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo,
            cls_branch=(128, 64),
            reg_branch=(
                (128, 64),  # offset
                (128, 64),  # depth
                (64, ),  # size
                (64, ),  # rot
                ()  # velo
            ),
            dir_branch=(64, ),
            attr_branch=(64, ),
            conv_cfg=None,
            norm_cfg=None,
            train_cfg=None,
            test_cfg=None,
            init_cfg=None):
        super(FreeMono3DHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_direction_classifier = use_direction_classifier
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.bbox_code_size = bbox_code_size
        self.group_reg_dims = list(group_reg_dims)
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        assert len(reg_branch) == len(group_reg_dims), 'The number of '\
            'element in reg_branch and group_reg_dims should be the same.'
        self.pred_velo = pred_velo
        self.pred_bbox2d = pred_bbox2d
        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self.dir_branch = dir_branch
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)
        self.pred_attrs = pred_attrs
        self.attr_background_label = -1
        self.num_attrs = num_attrs
        if self.pred_attrs:
            self.attr_background_label = num_attrs
            # self.loss_attr = build_loss(loss_attr)
            self.attr_branch = attr_branch

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        return conv_before_pred

    def _init_predictor(self):
        """初始化预测层的卷积层Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1, ) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels,
                                  1)
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1, ) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))
        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,
                conv_strides=(1, ) * len(self.dir_branch))
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)
        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1, ) * len(self.attr_branch))
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

    def forward(self, x):
        pass
    def forward_single(self,x):
        cls_feat = x
        reg_feat = x

        # 类别卷积
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # 保存特征
        clone_cls_feat = cls_feat.clone()
        # 预测类别
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score =self.conv_cls(clone_cls_feat)

        # 回归卷积
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # 检测框预测
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        dir_cls_pred = None
        if self.use_direction_classifier:
            # 方向分类预测
            clone_dir_feat = reg_feat.clone()
            for conv_dir_prev_layer in self.conv_dir_cls_prev:
                clone_dir_feat = conv_dir_prev_layer(clone_dir_feat)
            dir_cls_pred = self.conv_dir_cls(clone_dir_feat)
            
        attr_pred = None
        if self.pred_attrs:
            # 属性预测
            clone_attr_feat = reg_feat.clone()
            for conv_attr_prev_layer in self.conv_attr_prev:
                clone_attr_feat = conv_attr_prev_layer(clone_attr_feat)
            attr_pred = self.conv_attr(clone_attr_feat)

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat
    

