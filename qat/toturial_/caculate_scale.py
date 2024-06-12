import numpy as np
import copy


def KL(P, Q):
    out = np.sum(P * np.log(P / Q))
    return out


def maxq(value):
    dynamic_range = np.abs(value).max()
    scale = dynamic_range / 127.0
    return scale


def histogramq(value):
    hist, bins = np.histogram(value, 100)
    total = len(value)

    left, right = 0, len(hist)
    limit = 0.99  # 覆盖范围99%
    while True:
        nleft = left + 1
        nright = right - 1
        left_cover = hist[nleft:right].sum() / total
        right_cover = hist[left:nright].sum() / total
        # 范围缩紧
        if left_cover < limit and right_cover < limit:
            break
        if left_cover > right_cover:
            left += 1
        else:
            right -= 1

    low = bins[left]
    high = bins[right - 1]
    dynamic_range = max(abs(low), abs(high))
    scale = dynamic_range / 127.0
    return scale


def entropy(value, target_bin=128):
    # 计算最大绝对值
    amax = np.abs(value).max()
    # 计算直方图分布
    distribution, _ = np.histogram(value, bins=2048, range=(0, amax))
    # 遍历直方图分布，区间为[1:2048]
    distribution = distribution[1:]
    length = distribution.size 
    # 定义KL散度
    kl_divergence = np.zeros(length - target_bin)    
    # 遍历[128:2047]
    for threshold in range(target_bin, length):
        # slice分布，区间为[:threshold]
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])
        # 复制切分分布为：p
        p = sliced_nd_hist.copy()
        threshold_sum = sum(distribution[threshold:])
        # 边界外的组加到边界P[i-1]上，没有直接丢掉
        p[threshold-1] += threshold_sum
        is_nonzeros = (p != 0).astype(np.int64)
        # 合并bins，步长为：num_merged_bins=sliced_nd_hist.size // target_bin=16
        quantized_bins = np.zeros(target_bin, dtype=np.int64)    
        num_merged_bins = sliced_nd_hist.size // target_bin
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
        # 定义分布：q ，这里的size要和p分布一致，也就是和sliced_nd_hist分布一致
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        # 展开bins到q
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = -1 if j == target_bin - 1 else start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            q[start:stop] = float(quantized_bins[j]) / float(norm) if norm != 0 else q[start:stop]  
        # 归一化操作
        p = p / sum(p)
        q = q / sum(q)
        # 计算KL散度
        kl_divergence[threshold - target_bin] = KL(p, q)
    # 求出最小的 kl 散度
    min_kl_divergence = np.argmin(kl_divergence)
    # 求出最小 kl 散度对应的刻度
    threshold_value = min_kl_divergence + target_bin
    # 计算最终的threshold
    dynamic_range = (threshold_value + 0.5) * (amax / 2048)
    scale = dynamic_range / 127.0
    return scale


# int8截断，注意，-128丢掉了不要
def saturate(x):
    return np.clip(np.round(x), -127, +127) 


class Quant:
    def __init__(self, value):
        # 这里是对称量化,动态范围选取有多种方法，max/histogram/entropy等等
        # self.scale = maxq(value)
        # self.scale = histogramq(value)
        self.scale = entropy(value)

    def __call__(self, f):
        # 进行阶段
        return saturate(f / self.scale)


def Quant_Conv(x, w, b, iq, wq, oq=None):
    alpha = iq.scale * wq.scale
    out_int32 = iq(x) * wq(w)

    if oq is None:
        # float32 output
        return out_int32 * alpha + b
    else:
        # int8 quant output
        return saturate((out_int32 * alpha + b) / oq.scale)


if __name__ == '__main__':
    # x -> Q1 -> conv1 -> Q2 -> conv2 -> y
    np.random.seed(31)
    nelem = 1000
    # 生成随机权重、输入与偏置向量
    x = np.random.randn(nelem)
    weight1 = np.random.randn(nelem)
    bias1 = np.random.randn(nelem)

    # 计算第一层卷积计算的结果输出（fp32）
    t = x * weight1 + bias1
    weight2 = np.random.randn(nelem)
    bias2 = np.random.randn(nelem)

    # 计算第二层卷积计算的结果输出（fp32）
    y = t * weight2 + bias2
    # 分别对输入、权重以及中间层输出（也是下一层的输入）进行量化校准
    xQ  = Quant(x)
    w1Q = Quant(weight1)
    tQ  = Quant(t)
    w2Q = Quant(weight2)
    qt  = Quant_Conv(x, weight1, bias1, xQ, w1Q, tQ)
    # int8计算的结果输出
    y2  = Quant_Conv(qt, weight2, bias2, tQ, w2Q)
    # 计算量化计算的均方差
    y_diff = (np.abs(y - y2) ** 2).mean()
    print(f"ydiff mse error is: {y_diff}")

    '''
    max mse error        : 35.1663
    histogramq mse error : 8.6907
    entropy mse error    : 1.8590
    '''

