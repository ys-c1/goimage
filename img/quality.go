package img

import "math"

// 质量指标：结构相似度、过冲与差值统计，供核验与报表使用。

// SSIM 返回两张同尺寸灰度图的结构相似度（高斯窗 11x11，σ=1.5，C1/C2 取标准值）。
func SSIM(a, b *Gray) float64 {
	if a == nil || b == nil || a.W != b.W || a.H != b.H || a.W <= 0 || a.H <= 0 {
		return math.NaN()
	}
	const c1 = (0.01 * 255) * (0.01 * 255)
	const c2 = (0.03 * 255) * (0.03 * 255)
	n := a.W * a.H
	ax := make([]float64, n)
	bx := make([]float64, n)
	aa := make([]float64, n)
	bb := make([]float64, n)
	ab := make([]float64, n)
	for i := 0; i < n; i++ {
		x, y := float64(a.Pix[i]), float64(b.Pix[i])
		ax[i], bx[i] = x, y
		aa[i], bb[i], ab[i] = x*x, y*y, x*y
	}
	f := ssimFilter(a.W, a.H)
	ma, mb := f(ax), f(bx)
	va, vb, cov := f(aa), f(bb), f(ab)
	var sum float64
	for i := 0; i < n; i++ {
		siga := math.Max(0, va[i]-ma[i]*ma[i])
		sigb := math.Max(0, vb[i]-mb[i]*mb[i])
		sigab := cov[i] - ma[i]*mb[i]
		num := (2*ma[i]*mb[i] + c1) * (2*sigab + c2)
		den := (ma[i]*ma[i] + mb[i]*mb[i] + c1) * (siga + sigb + c2)
		sum += num / den
	}
	return sum / float64(n)
}

func ssimFilter(w, h int) func([]float64) []float64 {
	const r = 5
	k := make([]float64, 2*r+1)
	var ks float64
	for i := -r; i <= r; i++ {
		k[i+r] = math.Exp(-float64(i*i) / (2 * 1.5 * 1.5))
		ks += k[i+r]
	}
	for i := range k {
		k[i] /= ks
	}
	return func(src []float64) []float64 {
		tmp := make([]float64, len(src))
		dst := make([]float64, len(src))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var s float64
				for t := -r; t <= r; t++ {
					s += k[t+r] * src[y*w+clampInt(x+t, 0, w-1)]
				}
				tmp[y*w+x] = s
			}
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var s float64
				for t := -r; t <= r; t++ {
					s += k[t+r] * tmp[clampInt(y+t, 0, h-1)*w+x]
				}
				dst[y*w+x] = s
			}
		}
		return dst
	}
}

// Overshoot 汇总超出源图局部值域的像素统计。
type Overshoot struct {
	Pixels int // 过冲像素数
	Max    int // 单像素最大过冲幅度（灰度级）
	Sum    int // 过冲幅度总和（灰度级）
}

// OvershootStats 统计放大图 got 相对源图 src 的过冲。
//
// 判据取源图局部值域：对每个输出像素，按其中心映射到的源坐标取 3x3
// 源像素邻域的 [min,max]；got 落在域外即计为过冲，幅度为越界灰度级。
// 该判据比实现内部使用的 2x2 双线性足迹更宽，故能独立检验“过冲为零”。
func OvershootStats(got, src *Gray) Overshoot {
	var o Overshoot
	if got == nil || src == nil || got.W <= 0 || got.H <= 0 || src.W <= 0 || src.H <= 0 {
		return o
	}
	rx := float64(src.W) / float64(got.W)
	ry := float64(src.H) / float64(got.H)
	for y := 0; y < got.H; y++ {
		cy := clampInt(int(math.Floor((float64(y)+0.5)*ry)), 0, src.H-1)
		for x := 0; x < got.W; x++ {
			cx := clampInt(int(math.Floor((float64(x)+0.5)*rx)), 0, src.W-1)
			lo, hi := 255.0, 0.0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					p := float64(src.Pix[clampInt(cy+dy, 0, src.H-1)*src.W+clampInt(cx+dx, 0, src.W-1)])
					if p < lo {
						lo = p
					}
					if p > hi {
						hi = p
					}
				}
			}
			o.add(float64(got.Pix[y*got.W+x]), lo, hi)
		}
	}
	return o
}

// OvershootStatsFootprint 用与实现一致的双线性足迹判据统计过冲：
// 每个输出像素只与它中心映射处双线性插值实际触及的 2x2 源像素值域比较。
// 该判据最严格，直接检验“限幅到源值域”这一步是否生效。
func OvershootStatsFootprint(got, src *Gray) Overshoot {
	var o Overshoot
	if got == nil || src == nil || got.W <= 0 || got.H <= 0 || src.W <= 0 || src.H <= 0 {
		return o
	}
	rx := float64(src.W) / float64(got.W)
	ry := float64(src.H) / float64(got.H)
	for y := 0; y < got.H; y++ {
		v := (float64(y)+0.5)*ry - 0.5
		for x := 0; x < got.W; x++ {
			u := (float64(x)+0.5)*rx - 0.5
			lo, hi := footprintRange(src, u, v)
			o.add(float64(got.Pix[y*got.W+x]), lo, hi)
		}
	}
	return o
}

func (o *Overshoot) add(val, lo, hi float64) {
	var amt float64
	if val < lo {
		amt = lo - val
	} else if val > hi {
		amt = val - hi
	}
	if amt > 0 {
		a := int(math.Round(amt))
		if a < 1 {
			a = 1
		}
		o.Pixels++
		o.Sum += a
		if a > o.Max {
			o.Max = a
		}
	}
}

type Diff struct {
	MAE float64 // 平均绝对误差（灰度级）
	Max int     // 最大绝对误差（灰度级）
}

// DiffStats 返回 a 与 b 的逐像素差值统计；尺寸不一致返回 NaN MAE。
func DiffStats(a, b *Gray) Diff {
	if a == nil || b == nil || a.W != b.W || a.H != b.H || a.W <= 0 || a.H <= 0 {
		return Diff{MAE: math.NaN()}
	}
	var sum int
	var max int
	for i := range a.Pix {
		d := int(a.Pix[i]) - int(b.Pix[i])
		if d < 0 {
			d = -d
		}
		sum += d
		if d > max {
			max = d
		}
	}
	return Diff{MAE: float64(sum) / float64(len(a.Pix)), Max: max}
}
