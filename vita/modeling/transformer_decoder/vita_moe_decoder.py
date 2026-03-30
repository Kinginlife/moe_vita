"""
VITA Transformer Decoder with MoE in the last layer.
Based on vita_mask2former_transformer_decoder.py
"""
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .vita_mask2former_transformer_decoder import (
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
    MLP,
    _get_activation_fn
)
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from ..moe import MoELayer


class MoEFFNLayer(nn.Module):
    """FFN Layer with MoE for the last decoder layer."""

    def __init__(self, d_model, dim_feedforward=2048, num_experts=1,
                 router_dim=512, top_k=1, dropout=0.0,
                 normalize_before=False):
        super().__init__()
        self.moe = MoELayer(d_model, dim_feedforward, num_experts, router_dim, top_k, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def forward_post(self, tgt, routing_targets=None):
        tgt2, routing_loss = self.moe(tgt, routing_targets)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, routing_loss

    def forward_pre(self, tgt, routing_targets=None):
        tgt2 = self.norm(tgt)
        tgt2, routing_loss = self.moe(tgt2, routing_targets)
        tgt = tgt + self.dropout(tgt2)
        return tgt, routing_loss

    def forward(self, tgt, routing_targets=None):
        if self.normalize_before:
            return self.forward_pre(tgt, routing_targets)
        return self.forward_post(tgt, routing_targets)


@TRANSFORMER_DECODER_REGISTRY.register()
class VitaMoEMultiScaleMaskedTransformerDecoder(nn.Module):
    """VITA Decoder with MoE in the last FFN layer."""

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        # ================== 新增：MoE 权重兼容性映射 ==================
        # 遍历所有可能的层数
        for i in range(10): 
            old_k1 = prefix + f"transformer_ffn_layers.{i}.linear1.weight"
            moe_k1 = prefix + f"transformer_ffn_layers.{i}.moe.experts.0.linear1.weight"
            
            # 如果预训练字典里有旧的 linear，但没有 moe，我们主动帮它改名！
            if old_k1 in state_dict and moe_k1 not in state_dict:
                state_dict[moe_k1] = state_dict.pop(old_k1)
                state_dict[prefix + f"transformer_ffn_layers.{i}.moe.experts.0.linear1.bias"] = \
                    state_dict.pop(prefix + f"transformer_ffn_layers.{i}.linear1.bias")
                
                state_dict[prefix + f"transformer_ffn_layers.{i}.moe.experts.0.linear2.weight"] = \
                    state_dict.pop(prefix + f"transformer_ffn_layers.{i}.linear2.weight")
                state_dict[prefix + f"transformer_ffn_layers.{i}.moe.experts.0.linear2.bias"] = \
                    state_dict.pop(prefix + f"transformer_ffn_layers.{i}.linear2.bias")
                # 提示：你可能会在 log 里看到一两句关于 Router 的 unexpected_keys，那是正常的，因为 Router 本来就是新加的。
        # =============================================================
        if version is None or version < 2:
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        vita_last_layer_num: int,
        # MoE specific
        moe_num_experts: int = 1,
        moe_router_dim: int = 512,
        moe_top_k: int = 1,
        moe_num_layers: int = 1,
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            # Use MoE for last N layers, standard FFN for others
            if i >= self.num_layers - moe_num_layers:
                self.transformer_ffn_layers.append(
                    MoEFFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        num_experts=moe_num_experts,
                        router_dim=moe_router_dim,
                        top_k=moe_top_k,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
            else:
                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.vita_last_layer_num = vita_last_layer_num

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["vita_last_layer_num"] = cfg.MODEL.VITA.LAST_LAYER_NUM

        # MoE config
        ret["moe_num_experts"] = cfg.MOE.NUM_EXPERTS
        ret["moe_router_dim"] = cfg.MOE.ROUTER_DIM
        ret["moe_top_k"] = cfg.MOE.TOP_K
        ret["moe_num_layers"] = cfg.MOE.NUM_MOE_LAYERS

        return ret

    def forward(self, x, mask_features, clip_mask_features, mask=None, routing_targets=None):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        frame_queries = []
        predictions_class = []
        predictions_mask = []
        routing_losses = []

        outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN: check if this layer uses MoE
            if isinstance(self.transformer_ffn_layers[i], MoEFFNLayer):
                output, routing_loss = self.transformer_ffn_layers[i](output, routing_targets)
                if routing_loss is not None:
                    routing_losses.append(routing_loss)
            else:
                output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            frame_queries.append(frame_query)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }

        if routing_losses:
            out['routing_loss'] = sum(routing_losses) / len(routing_losses)

        num_layer = self.vita_last_layer_num if self.training else 1
        frame_queries = torch.stack(frame_queries[-num_layer:])

        return out, frame_queries, clip_mask_features

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, decoder_output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

