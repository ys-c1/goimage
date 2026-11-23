package img

import (
	"fmt"
	"math"
)

// grayLevels 是灰度级数，目标分布与映射均定义在 [0,255] 上。
const grayLevels = 256

// FieldError 报出非法输入的字段名，便于调用方定位。
type FieldError struct {
	Field string
	Msg   string
}

func (e *FieldError) Error() string {
	return fmt.Sprintf("字段 %s: %s", e.Field, e.Msg)
}

// MatchResult 保存一次最优直方图匹配的全部结果。
type MatchResult struct {
	Out        *Gray            // 匹配后的图像
	LUT        [grayLevels]uint8 // 单调不减的灰度映射表
	Centroids  []float64        // 目标分布最优标量量化的质心
	Iterations int              // Lloyd 迭代次数
	Converged  bool             // 是否收敛到不动点
	W1         float64          // 匹配结果与目标的推土机距离（W1，精确值）
	W1Bound    float64          // W1 的理论上界
	UniformW1  float64          // 基线一：均匀分箱的 W1
	LinearW1   float64          // 基线二：线性映射的 W1
}

// MatchHistogram 把源图灰度分布扳到目标分布。
//
// 步骤：
//  1. 对目标分布在 [0,255] 上做最优标量量化（Lloyd-Max）：迭代
//     “边界=相邻质心中点 → 质心=箱内条件均值”，收敛到不动点。
//     收敛判据：单次迭代所有质心的最大位移 < 1e-9。Lloyd 迭代使
//     量化失真 D=Σp(v)(v-c(v))² 单调不增且有下界 0，故必收敛；
//     不动点处边界恰为质心中点、质心恰为箱内均值。
//  2. 源图施加单调映射：v → 目标CDF逆(源中值CDF(v))，再吸附到最近
//     质心。CDF 逆映射单调不减，吸附到有序质心仍单调不减，故 LUT
//     单调不减。
//  3. 上界：设 m 为未吸附的理想映射，则 W1(输出,目标) ≤
//     W1(输出,m推前) + W1(m推前,目标)，两项均精确计算。
//
// target 必须为 256 维、非负、有限、总和为正。
func MatchHistogram(g *Gray, target []float64, bins int) (*MatchResult, error) {
	if g == nil || len(g.Pix) == 0 {
		return nil, &FieldError{Field: "Image", Msg: "图为空"}
	}
	if bins < 1 {
		return nil, &FieldError{Field: "Bins", Msg: "分箱数必须为正"}
	}
	if len(target) != grayLevels {
		return nil, &FieldError{Field: "Target", Msg: "分布长度必须为 256"}
	}
	sum := 0.0
	for _, v := range target {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, &FieldError{Field: "Target", Msg: "分布含负值或非有限值"}
		}
		sum += v
	}
	if sum <= 0 {
		return nil, &FieldError{Field: "Target", Msg: "分布总和必须为正"}
	}
	var t [grayLevels]float64
	for i, v := range target {
		t[i] = v / sum
	}

	centroids, iters, conv := lloydMax(&t, bins)

	// 源直方图
	var hs [grayLevels]float64
	for _, v := range g.Pix {
		hs[v]++
	}
	n := float64(len(g.Pix))

	// 目标累积分布
	var cumT [grayLevels]float64
	acc := 0.0
	for i := 0; i < grayLevels; i++ {
		acc += t[i]
		cumT[i] = acc
	}

	// 单调映射：中值 CDF 匹配后吸附到最近质心
	var lut [grayLevels]uint8
	m := make([]float64, grayLevels)
	cumS := 0.0
	u := 0
	for v := 0; v < grayLevels; v++ {
		f := (cumS + 0.5*hs[v]) / n // 中值 CDF，单像素时取到目标中位数
		cumS += hs[v]
		for u < grayLevels-1 && cumT[u] < f {
			u++
		}
		m[v] = float64(u)
		best := 0
		for i := 1; i < len(centroids); i++ {
			if math.Abs(centroids[i]-m[v]) < math.Abs(centroids[best]-m[v]) {
				best = i
			}
		}
		lut[v] = uint8(math.Round(centroids[best]))
	}

	out, err := New(g.W, g.H)
	if err != nil {
		return nil, err
	}
	for i, v := range g.Pix {
		out.Pix[i] = lut[v]
	}

	// 输出与理想映射的直方图
	var hOut, hIdeal [grayLevels]float64
	for v := 0; v < grayLevels; v++ {
		hOut[lut[v]] += hs[v] / n
		hIdeal[int(m[v])] += hs[v] / n
	}
	w1 := wasserstein1(&hOut, &t)
	bound := wasserstein1(&hOut, &hIdeal) + wasserstein1(&hIdeal, &t)

	return &MatchResult{
		Out:        out,
		LUT:        lut,
		Centroids:  centroids,
		Iterations: iters,
		Converged:  conv,
		W1:         w1,
		W1Bound:    bound,
		UniformW1:  wasserstein1(uniformBinHist(&hs, n, bins), &t),
		LinearW1:   wasserstein1(linearMapHist(&hs, n, &t), &t),
	}, nil
}

// lloydMax 对 [0,255] 上的分布 t 做 k 点最优标量量化。
// 返回质心、迭代次数与是否收敛（质心最大位移 < 1e-9）。
func lloydMax(t *[grayLevels]float64, k int) (centroids []float64, iters int, converged bool) {
	const (
		tol     = 1e-9
		maxIter = 10000
	)
	// 初始化：目标分布的等概率分位点
	var cum [grayLevels]float64
	acc := 0.0
	for i := 0; i < grayLevels; i++ {
		acc += t[i]
		cum[i] = acc
	}
	centroids = make([]float64, k)
	for i := 0; i < k; i++ {
		q := (float64(i) + 0.5) / float64(k)
		v := 0
		for v < grayLevels-1 && cum[v] < q {
			v++
		}
		centroids[i] = float64(v)
	}
	for it := 1; it <= maxIter; it++ {
		iters = it
		// 边界 = 相邻质心中点
		b := make([]float64, k+1)
		b[0] = -0.5
		b[k] = float64(grayLevels) - 0.5
		for i := 1; i < k; i++ {
			b[i] = (centroids[i-1] + centroids[i]) / 2
		}
		// 质心 = 箱内条件均值
		sums := make([]float64, k)
		ws := make([]float64, k)
		cell := 0
		for v := 0; v < grayLevels; v++ {
			for cell < k-1 && float64(v) >= b[cell+1] {
				cell++
			}
			sums[cell] += t[v] * float64(v)
			ws[cell] += t[v]
		}
		nc := make([]float64, k)
		shift := 0.0
		for i := 0; i < k; i++ {
			if ws[i] > 0 {
				nc[i] = sums[i] / ws[i]
			} else {
				nc[i] = centroids[i] // 空箱保留原质心
			}
			if i > 0 && nc[i] < nc[i-1] {
				nc[i] = nc[i-1] // 保持有序
			}
			if d := math.Abs(nc[i] - centroids[i]); d > shift {
				shift = d
			}
		}
		centroids = nc
		if shift < tol {
			converged = true
			break
		}
	}
	return centroids, iters, converged
}

// wasserstein1 计算 [0,255] 整数格点上两分布的 W1（推土机距离）：
// W1 = Σ_k |F_p(k) - F_q(k)|。
func wasserstein1(p, q *[grayLevels]float64) float64 {
	d := 0.0
	acc := 0.0
	for k := 0; k < grayLevels-1; k++ {
		acc += p[k] - q[k]
		d += math.Abs(acc)
	}
	return d
}

// uniformBinHist 基线一：均匀分箱，把每级灰度压到所在等宽箱的中心。
func uniformBinHist(hs *[grayLevels]float64, n float64, bins int) *[grayLevels]float64 {
	var h [grayLevels]float64
	w := float64(grayLevels) / float64(bins)
	for v := 0; v < grayLevels; v++ {
		if hs[v] == 0 {
			continue
		}
		c := int(float64(v) / w)
		if c >= bins {
			c = bins - 1
		}
		center := (float64(c)+0.5)*w - 0.5
		if center < 0 {
			center = 0
		}
		lv := int(math.Round(center))
		if lv > grayLevels-1 {
			lv = grayLevels - 1
		}
		h[lv] += hs[v] / n
	}
	return &h
}

// linearMapHist 基线二：线性映射，匹配源与目标的均值和标准差后截断。
func linearMapHist(hs *[grayLevels]float64, n float64, t *[grayLevels]float64) *[grayLevels]float64 {
	meanS, stdS := histMeanStd(hs)
	meanT, stdT := histMeanStd(t)
	a := 0.0
	if stdS > 0 {
		a = stdT / stdS
	}
	b := meanT - a*meanS
	var h [grayLevels]float64
	for v := 0; v < grayLevels; v++ {
		if hs[v] == 0 {
			continue
		}
		lv := int(math.Round(a*float64(v) + b))
		if lv < 0 {
			lv = 0
		}
		if lv > grayLevels-1 {
			lv = grayLevels - 1
		}
		h[lv] += hs[v] / n
	}
	return &h
}

func histMeanStd(h *[grayLevels]float64) (mean, std float64) {
	for v := 0; v < grayLevels; v++ {
		mean += h[v] * float64(v)
	}
	for v := 0; v < grayLevels; v++ {
		d := float64(v) - mean
		std += h[v] * d * d
	}
	return mean, math.Sqrt(std)
}
