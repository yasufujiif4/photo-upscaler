"""CodeFormer と顔検出（RetinaFace mobilenet0.25）を ONNX に変換する。

必要なもの:
  - CodeFormer のソース: git clone --depth 1 https://github.com/sczhou/CodeFormer
  - 重み: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
          https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth
  - pip install torch onnx numpy

使い方:
  python tools/export_codeformer_onnx.py <CodeFormerソースのフォルダ> <codeformer.pth> <detection_mobilenet0.25_Final.pth> <出力フォルダ>

出力:
  codeformer.onnx  入力 input [1,3,512,512]（RGB, -1〜1）, w [1]（0=きれいさ優先〜1=本人らしさ優先）
                   出力 output [1,3,512,512]（-1〜1）
                   重みは float16 で保存し、読み込み時に float32 に戻す（ファイルを半分にするため）
  retinaface.onnx  入力 input [1,3,H,W]（BGR, 0〜255 から平均 (104,117,123) を引いたもの）
                   出力 loc, conf, landms
"""
import importlib.util
import sys
import types

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import helper, numpy_helper, TensorProto


def stub_basicsr():
    """basicsr の他の部分（cv2 などが必要）を読み込まずに、アーキテクチャだけ使う"""
    import logging
    utils = types.ModuleType('basicsr.utils')
    utils.get_root_logger = lambda *a, **k: logging.getLogger('basicsr')
    registry = types.ModuleType('basicsr.utils.registry')

    class Reg:
        def register(self, *a, **k):
            return (lambda x: x) if not a or not callable(a[0]) else a[0]
    registry.ARCH_REGISTRY = Reg()
    for name, mod in [('basicsr', types.ModuleType('basicsr')), ('basicsr.utils', utils),
                      ('basicsr.utils.registry', registry), ('basicsr.archs', types.ModuleType('basicsr.archs'))]:
        sys.modules[name] = mod


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class CodeFormerONNX(torch.nn.Module):
    """推論用の包み。w を入力にし、ONNX にしやすい形に書き換える（計算は元と同じ）"""

    def __init__(self, net, arch):
        super().__init__()
        self.net = net
        self.arch = arch

    def forward(self, x, w):
        net = self.net
        # 入力は 512 固定なので、何番目の層がどの大きさかは決まっている（トレース中に shape を文字にしない）
        enc_at = {net.fuse_encoder_block[f]: f for f in net.connect_list}
        gen_at = {net.fuse_generator_block[f]: f for f in net.connect_list}
        enc_feat_dict = {}
        for i, block in enumerate(net.encoder.blocks):
            x = block(x)
            if i in enc_at:
                enc_feat_dict[enc_at[i]] = x
        lq_feat = x
        pos_emb = net.position_emb.unsqueeze(1)
        query_emb = net.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        for layer in net.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        logits = net.idx_pred_layer(query_emb).permute(1, 0, 2)  # b(hw)n
        idx = torch.argmax(logits, dim=2)                          # softmax→topk と同じ
        quant = F.embedding(idx, net.quantize.embedding.weight)    # b(hw)c
        quant = quant.view(1, 16, 16, 256).permute(0, 3, 1, 2)
        quant = self.arch.adaptive_instance_normalization(quant, lq_feat)
        x = quant
        for i, block in enumerate(net.generator.blocks):
            x = block(x)
            if i in gen_at:
                f = gen_at[i]
                # 元は w>0 のときだけ足す。w=0 なら足す量が 0 なので、いつも足しても同じ
                x = net.fuse_convs_dict[f](enc_feat_dict[f], x, w)
        return x


def weights_to_fp16(model, min_size=1024):
    """大きな float32 の重みを float16 で保存し、Cast で float32 に戻す"""
    g = model.graph
    new_inits, cast_nodes = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= min_size:
            h = numpy_helper.from_array(arr.astype(np.float16), init.name + '_fp16')
            new_inits.append(h)
            cast_nodes.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(cast_nodes + nodes)
    return model


def main(src, cf_pth, det_pth, outdir):
    sys.path.insert(0, src)
    stub_basicsr()
    load_module('basicsr.archs.vqgan_arch', f'{src}/basicsr/archs/vqgan_arch.py')
    arch = load_module('basicsr.archs.codeformer_arch', f'{src}/basicsr/archs/codeformer_arch.py')

    net = arch.CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                          connect_list=['32', '64', '128', '256']).eval()
    ckpt = torch.load(cf_pth, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['params_ema'])
    wrapped = CodeFormerONNX(net, arch).eval()

    x = torch.rand(1, 3, 512, 512) * 2 - 1
    w = torch.tensor([0.7])
    with torch.no_grad():
        ref = net(x, w=0.7, adain=True)[0]
        got = wrapped(x, w)
    print('codeformer wrapper max diff', float((ref - got).abs().max()))

    path = f'{outdir}/codeformer.onnx'
    torch.onnx.export(wrapped, (x, w), path, input_names=['input', 'w'], output_names=['output'],
                      opset_version=17, dynamo=False)
    m = onnx.load(path)
    onnx.save(weights_to_fp16(m), path)
    print('saved', path)

    export_retinaface(src, det_pth, outdir)


def export_retinaface(src, det_pth, outdir):
    """顔検出（RetinaFace mobilenet0.25）。torchvision や cv2 を使わずに同じ作りを組み立てる"""
    rf = load_module('retinaface_net', f'{src}/facelib/detection/retinaface/retinaface_net.py')

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            m = rf.MobileNetV1()
            self.stage1, self.stage2, self.stage3 = m.stage1, m.stage2, m.stage3

        def forward(self, x):
            a = self.stage1(x); b = self.stage2(a); c = self.stage3(b)
            return [a, b, c]

    class RetinaFaceONNX(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = Body()
            self.fpn = rf.FPN([64, 128, 256], 64)
            self.ssh1, self.ssh2, self.ssh3 = rf.SSH(64, 64), rf.SSH(64, 64), rf.SSH(64, 64)
            self.ClassHead = rf.make_class_head(fpn_num=3, inchannels=64)
            self.BboxHead = rf.make_bbox_head(fpn_num=3, inchannels=64)
            self.LandmarkHead = rf.make_landmark_head(fpn_num=3, inchannels=64)

        def forward(self, x):
            fpn = self.fpn(self.body(x))
            feats = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
            loc = torch.cat([self.BboxHead[i](f) for i, f in enumerate(feats)], dim=1)
            conf = torch.cat([self.ClassHead[i](f) for i, f in enumerate(feats)], dim=1)
            landms = torch.cat([self.LandmarkHead[i](f) for i, f in enumerate(feats)], dim=1)
            return loc, F.softmax(conf, dim=-1), landms

    net = RetinaFaceONNX().eval()
    sd = torch.load(det_pth, map_location='cpu', weights_only=False)
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print('retinaface missing', missing, 'unexpected', unexpected)
    x = torch.zeros(1, 3, 640, 640)
    path = f'{outdir}/retinaface.onnx'
    torch.onnx.export(net, (x,), path, input_names=['input'], output_names=['loc', 'conf', 'landms'],
                      dynamic_axes={'input': {2: 'h', 3: 'w'}, 'loc': {1: 'n'}, 'conf': {1: 'n'}, 'landms': {1: 'n'}},
                      opset_version=17, dynamo=False)
    print('saved', path)


if __name__ == '__main__':
    if len(sys.argv) > 5 and sys.argv[5] == 'det-only':
        export_retinaface(*sys.argv[1:2], sys.argv[3], sys.argv[4])
    else:
        main(*sys.argv[1:5])
