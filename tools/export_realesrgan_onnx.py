"""Real-ESRGAN x4plus の .pth（PyTorch）を、PyTorch なしで ONNX に変換する。

使い方:
  python tools/export_realesrgan_onnx.py RealESRGAN_x4plus.pth models/realesrgan-x4plus.onnx

重みの入手元（公式）:
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
入力: float32 [1,3,H,W]（RGB, 0〜1） / 出力: float32 [1,3,4H,4W]
"""
import pickle
import sys
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DTYPES = {
    'FloatStorage': np.float32, 'HalfStorage': np.float16, 'DoubleStorage': np.float64,
    'LongStorage': np.int64, 'IntStorage': np.int32,
}


def load_pth(path):
    """torch.save の zip 形式を numpy 配列の dict として読む"""
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split('/')[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                def rebuild(storage, offset, size, stride, *args):
                    if not size:
                        return storage[offset].copy()
                    item = storage.itemsize
                    return np.lib.stride_tricks.as_strided(
                        storage[offset:], shape=size, strides=[s * item for s in stride]).copy()
                return rebuild
            if module == 'torch' and name.endswith('Storage'):
                return name
            if module == 'collections' and name == 'OrderedDict':
                import collections
                return collections.OrderedDict
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, storage_type, key, _location, _numel = pid
            dtype = DTYPES[storage_type if isinstance(storage_type, str) else storage_type.__name__]
            return np.frombuffer(zf.read(f'{prefix}/data/{key}'), dtype=dtype)

    return Unpickler(zf.open(f'{prefix}/data.pkl')).load()


def build(sd, num_block=23):
    nodes, inits = [], []
    counter = [0]

    def name(p):
        counter[0] += 1
        return f'{p}_{counter[0]}'

    def conv(x, key, act=False):
        w, b = key + '.weight', key + '.bias'
        inits.append(numpy_helper.from_array(sd[w].astype(np.float32), w))
        inits.append(numpy_helper.from_array(sd[b].astype(np.float32), b))
        y = name('conv')
        nodes.append(helper.make_node('Conv', [x, w, b], [y], pads=[1, 1, 1, 1], kernel_shape=[3, 3]))
        if act:
            z = name('lrelu')
            nodes.append(helper.make_node('LeakyRelu', [y], [z], alpha=0.2))
            return z
        return y

    def const(v):
        n = name('c')
        inits.append(numpy_helper.from_array(np.array(v, dtype=np.float32), n))
        return n

    k02 = const(0.2)

    def op(t, *ins, **kw):
        y = name(t.lower())
        nodes.append(helper.make_node(t, list(ins), [y], **kw))
        return y

    def rdb(x, p):
        x1 = conv(x, p + '.conv1', True)
        x2 = conv(op('Concat', x, x1, axis=1), p + '.conv2', True)
        x3 = conv(op('Concat', x, x1, x2, axis=1), p + '.conv3', True)
        x4 = conv(op('Concat', x, x1, x2, x3, axis=1), p + '.conv4', True)
        x5 = conv(op('Concat', x, x1, x2, x3, x4, axis=1), p + '.conv5')
        return op('Add', op('Mul', x5, k02), x)

    def up2(x):
        scales = const([1, 1, 2, 2])
        return op('Resize', x, '', scales, mode='nearest', coordinate_transformation_mode='asymmetric',
                  nearest_mode='floor')

    feat = conv('input', 'conv_first')
    h = feat
    for i in range(num_block):
        p = f'body.{i}'
        o = rdb(rdb(rdb(h, p + '.rdb1'), p + '.rdb2'), p + '.rdb3')
        h = op('Add', op('Mul', o, k02), h)
    feat = op('Add', feat, conv(h, 'conv_body'))
    feat = conv(up2(feat), 'conv_up1', True)
    feat = conv(up2(feat), 'conv_up2', True)
    out = conv(conv(feat, 'conv_hr', True), 'conv_last')
    nodes.append(helper.make_node('Identity', [out], ['output']))

    graph = helper.make_graph(
        nodes, 'realesrgan_x4plus',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 'h', 'w'])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 'H', 'W'])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = load_pth(src)
    sd = ckpt.get('params_ema') or ckpt.get('params') or ckpt
    used = build(sd)
    onnx.save(used, dst)
    print('saved', dst)
