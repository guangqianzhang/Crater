import torch.nn as nn
import torch
from freemono3dhead import FreeMono3DHead
from mmcv.cnn import Scale, normal_init
from mmdet.core.bbox.builder import build_bbox_coder

from mmdet.core import multi_apply
from mmcv.runner import force_fp32
INF = 1e8
class bbox_head(FreeMono3DHead):
    def __init__(self, 
                 regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                                 (384, INF)),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 centerness_on_reg=True,
                 centerness_alpha=2.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_dir=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_attr=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=9),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 centerness_branch=(64, ),
                 init_cfg=None,
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.centerness_alpha = centerness_alpha
        self.centerness_branch = centerness_branch
        super(bbox_head, self).__init__(loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        
        bbox_coder['code_size'] = self.bbox_code_size
        self.bbox_coder = build_bbox_coder(bbox_coder)



    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 3  # only for offset, depth and size regression
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)[:5]
    def forward_single(self, x, scale, stride):
        # 系列卷积后得到特征图
        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat = \
            super().forward_single(x)
        # 计算中心点坐标
        if self.centerness_on_reg:
            clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness= self.conv_centerness(clone_reg_feat)
        else:
            clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)

        bbox_pred = self.bbox_coder.decode(
            bbox_pred,
            scale,
            stride,
            self.training, cls_score)
        return cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds', 'attr_preds',
                  'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   attr_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(centernesses) == len(attr_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.pred_attrs:
                attr_pred_list = [
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                attr_pred_list, centerness_pred_list, mlvl_points, input_meta,
                cfg, rescale)
            result_list.append(det_bboxes)
        return result_list
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels and attributes.
        """
        view = np.array(input_meta['cam2img'])
        scale_factor = input_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_scores, bbox_preds, dir_cls_preds,
                              attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred = bbox_pred[:, :self.bbox_code_size]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
            # change the offset to actual center predictions
            bbox_pred[:, :2] = points - bbox_pred[:, :2]
            if rescale:
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)
            pred_center2d = bbox_pred[:, :3].clone()
            bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        # change local yaw to global yaw for 3D nms
        cam2img = mlvl_centers2d.new_zeros((4, 4))
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = input_meta['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)
        if not self.pred_attrs:
            attrs = None

        return bboxes, scores, labels, attrs
