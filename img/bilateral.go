package img

import (
	"errors"
	"fmt"
	"math"
)

var (
	ErrBilateralImage  = errors.New("图像为空")
	ErrBilateralRadius = errors.New("半径必须为正且不得超过反解上限")
	ErrBilateralSigma  = errors.New("sigma 必须为正的有限值")
	ErrBilateralRatio  = errors.New("保留率或噪声抑制倍数非法")
	ErrBilateralRegion = errors.New("统计区域非法")
)

// BilateralParams 双边滤波参数。Radius 给零、Sigma 取非法值、图为空都会被拦下，
// 错误信息中带出字段名。
type BilateralParams struct {
	Radius         int     // 空间半径，>0，且不得超过由 NoiseReduction 反解的上限
	SigmaS         float64 // 空间 sigma，正的有限值
	SigmaR         float64 // 灰度 sigma，正的有限值（可由 SolveSigmaR 反解）
	NoiseReduction float64 // 目标噪声抑制倍数；>1 时启用半径上限校验
}

func fieldErr(name string, err error) error {
	return fmt.Errorf("bilateral: 字段 %s: %w", name, err)
}

func validSigma(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func (p BilateralParams) validate(g *Gray) error {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return fieldErr("Image", ErrBilateralImage)
	}
	if p.Radius <= 0 {
		return fieldErr("Radius", ErrBilateralRadius)
	}
	if !validSigma(p.SigmaS) {
		return fieldErr("SigmaS", ErrBilateralSigma)
	}
	if !validSigma(p.SigmaR) {
		return fieldErr("SigmaR", ErrBilateralSigma)
	}
	if p.NoiseReduction > 1 {
		rmax, _, err := SolveRadius(p.NoiseReduction)
		if err != nil {
			return fieldErr("NoiseReduction", err)
		}
		if p.Radius > rmax {
			return fieldErr("Radius", fmt.Errorf("%w: %d 超过反解值 %d", ErrBilateralRadius, p.Radius, rmax))
		}
	}
	return nil
}

// SolveSigmaR 由保边条件反解灰度 sigma。
//
// 联合权重 w(d,ΔI) = exp(-d²/2σs²)·exp(-ΔI²/2σr²)。
// 沿边缘方向（灰度差 ΔI 方向）对灰度权重 wr 求导：
//
//	wr'(ΔI)  = -(ΔI/σr²)·wr(ΔI)
//	wr''(ΔI) = (ΔI²/σr⁴ - 1/σr²)·wr(ΔI)
//
// 令 wr''=0 得拐点 ΔI*=σr：权重对灰度差的区分度在 ΔI=σr 处最强。
// 高度为 h 的阶跃边经对称核滤波后边缘梯度 G(σr)=h·(1-r)/(1+r)，
// 其中 r=exp(-h²/2σr²) 为跨边泄漏权重。dG/dσr<0，衰减随 σr 单调上升，
// 故保边约束 G/h≥ρ 在等号处给出去噪最强的可行解：
//
//	(1-r)/(1+r) ≥ ρ  ⇒  r ≤ (1-ρ)/(1+ρ)  ⇒  σr ≤ h/√(2·ln((1+ρ)/(1-ρ)))
//
// 取等号即为所求。返回 sigma、反解过程与错误。
func SolveSigmaR(edgeContrast, preserveRatio float64) (float64, []string, error) {
	if !validSigma(edgeContrast) {
		return 0, nil, fieldErr("EdgeContrast", ErrBilateralSigma)
	}
	if preserveRatio <= 0 || preserveRatio >= 1 || math.IsNaN(preserveRatio) {
		return 0, nil, fieldErr("PreserveRatio", ErrBilateralRatio)
	}
	den := math.Sqrt(2 * math.Log((1+preserveRatio)/(1-preserveRatio)))
	sigma := edgeContrast / den
	steps := []string{
		"联合权重 w(d,ΔI) = exp(-d²/2σs²)·exp(-ΔI²/2σr²)，空间项与灰度项分离",
		"沿边缘方向对灰度权重求导：wr'(ΔI) = -(ΔI/σr²)·wr(ΔI)",
		"二阶导 wr''(ΔI) = (ΔI²/σr⁴ - 1/σr²)·wr(ΔI)，令 wr''=0 得拐点 ΔI*=σr（区分度最强处）",
		"阶跃边滤波后梯度 G(σr)=h·(1-r)/(1+r)，r=exp(-h²/2σr²)；dG/dσr<0，衰减随 σr 单调上升",
		"保边约束 G/h≥ρ 取等号：(1-r)/(1+r)=ρ ⇒ r=(1-ρ)/(1+ρ) ⇒ σr=h/√(2·ln((1+ρ)/(1-ρ)))",
		fmt.Sprintf("代入 h=%.4g, ρ=%.4g：σr = %.4g/√(2·ln(%.4g/%.4g)) = %.6g",
			edgeContrast, preserveRatio, edgeContrast, 1+preserveRatio, 1-preserveRatio, sigma),
	}
	return sigma, steps, nil
}

// SolveRadius 由目标噪声抑制倍数反解半径。
//
// 平坦区噪声经归一化权重平均后方差比 = Σwᵢ²（Σwᵢ=1）。二维高斯空间核
// 按惯例取截断半径 R=2σs，其方差比 ≈ 1/(4πσs²) = 1/(πR²)。要求方差
// 下降 k 倍：1/(πR²) ≤ 1/k ⇒ R ≥ √(k/π)。反解值 R*=⌈√(k/π)⌉ 为恰好
// 达标的半径；实际半径不得超过 R*，否则平滑超出目标（过度平滑）。
func SolveRadius(noiseReduction float64) (int, []string, error) {
	if noiseReduction <= 1 || math.IsNaN(noiseReduction) || math.IsInf(noiseReduction, 0) {
		return 0, nil, fieldErr("NoiseReduction", ErrBilateralRatio)
	}
	exact := math.Sqrt(noiseReduction / math.Pi)
	r := int(math.Ceil(exact))
	steps := []string{
		"平坦区方差比 = Σwᵢ²（归一化权重，Σwᵢ=1）",
		"二维高斯空间核（R=2σs）方差比 ≈ 1/(4πσs²) = 1/(πR²)",
		"要求下降 k 倍：1/(πR²) ≤ 1/k ⇒ R ≥ √(k/π)，反解值 R*=⌈√(k/π)⌉，实际半径不得超过 R*",
		fmt.Sprintf("代入 k=%.4g：√(k/π)=%.6g，R*=%d", noiseReduction, exact, r),
	}
	return r, steps, nil
}

// Bilateral 双边滤波。不改写输入，返回新图。
func Bilateral(g *Gray, p BilateralParams) (*Gray, error) {
	if err := p.validate(g); err != nil {
		return nil, err
	}
	out, err := New(g.W, g.H)
	if err != nil {
		return nil, err
	}
	r := p.Radius
	inv2ss := 1 / (2 * p.SigmaS * p.SigmaS)
	inv2sr := 1 / (2 * p.SigmaR * p.SigmaR)
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			c := float64(g.Pix[y*g.W+x])
			var wsum, acc float64
			for dy := -r; dy <= r; dy++ {
				yy := y + dy
				if yy < 0 || yy >= g.H {
					continue
				}
				for dx := -r; dx <= r; dx++ {
					xx := x + dx
					if xx < 0 || xx >= g.W {
						continue
					}
					v := float64(g.Pix[yy*g.W+xx])
					d := v - c
					w := math.Exp(-float64(dx*dx+dy*dy)*inv2ss-d*d*inv2sr)
					wsum += w
					acc += w * v
				}
			}
			q := math.Round(acc / wsum)
			if q < 0 {
				q = 0
			}
			if q > 255 {
				q = 255
			}
			out.Pix[y*g.W+x] = uint8(q)
		}
	}
	return out, nil
}

func maxGradient(g *Gray) float64 {
	var m float64
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			v := float64(g.Pix[y*g.W+x])
			if x+1 < g.W {
				if d := math.Abs(float64(g.Pix[y*g.W+x+1]) - v); d > m {
					m = d
				}
			}
			if y+1 < g.H {
				if d := math.Abs(float64(g.Pix[(y+1)*g.W+x]) - v); d > m {
					m = d
				}
			}
		}
	}
	return m
}

// EdgeRetention 边缘梯度保留率：滤波后全图最大梯度 / 滤波前全图最大梯度。
// 与 FlatVariance 分开独立验证。纯色图（原梯度为零）视为完全保留，返回 1。
func EdgeRetention(before, after *Gray) (float64, error) {
	if before == nil || after == nil || len(before.Pix) == 0 || len(after.Pix) == 0 {
		return 0, fieldErr("Image", ErrBilateralImage)
	}
	if before.W != after.W || before.H != after.H {
		return 0, fieldErr("Image", errors.New("两图尺寸不一致"))
	}
	gb := maxGradient(before)
	if gb == 0 {
		return 1, nil
	}
	return maxGradient(after) / gb, nil
}

// FlatVariance 平坦区灰度方差，区域为 [x0,x1)×[y0,y1)。
func FlatVariance(g *Gray, x0, y0, x1, y1 int) (float64, error) {
	if g == nil || len(g.Pix) == 0 {
		return 0, fieldErr("Image", ErrBilateralImage)
	}
	if x0 < 0 || y0 < 0 || x1 > g.W || y1 > g.H || x1 <= x0 || y1 <= y0 {
		return 0, fieldErr("Region", ErrBilateralRegion)
	}
	var sum, sum2 float64
	var n int
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			v := float64(g.Pix[y*g.W+x])
			sum += v
			sum2 += v * v
			n++
		}
	}
	mean := sum / float64(n)
	return sum2/float64(n) - mean*mean, nil
}

// VarianceDrop 平坦区方差降幅 = 滤波前方差 / 滤波后方差。
// 原方差为零时：滤波后仍为零返回 1，否则返回 0。
func VarianceDrop(before, after *Gray, x0, y0, x1, y1 int) (float64, error) {
	vb, err := FlatVariance(before, x0, y0, x1, y1)
	if err != nil {
		return 0, err
	}
	va, err := FlatVariance(after, x0, y0, x1, y1)
	if err != nil {
		return 0, err
	}
	if vb == 0 {
		if va == 0 {
			return 1, nil
		}
		return 0, nil
	}
	if va == 0 {
		return math.Inf(1), nil
	}
	return vb / va, nil
}

// DenoiseReport 收尾报告：保留率、方差降幅与反解过程。
type DenoiseReport struct {
	SigmaR        float64  // 反解出的灰度 sigma
	Radius        int      // 反解出的半径（实际使用值，等于反解上限）
	SigmaS        float64  // 空间 sigma（取 Radius/2）
	Derivation    []string // 反解过程
	EdgeRetention float64  // 边缘梯度保留率
	VarianceDrop  float64  // 平坦区方差降幅
}

// Denoise 反解参数 → 双边滤波 → 分开验证两条指标，返回结果与报告。
// flat 为平坦区 [x0,y0,x1,y1)，传 nil 则跳过方差验证。
func Denoise(g *Gray, edgeContrast, preserveRatio, noiseReduction float64, flat []int) (*Gray, *DenoiseReport, error) {
	sigmaR, s1, err := SolveSigmaR(edgeContrast, preserveRatio)
	if err != nil {
		return nil, nil, err
	}
	radius, s2, err := SolveRadius(noiseReduction)
	if err != nil {
		return nil, nil, err
	}
	p := BilateralParams{
		Radius:         radius,
		SigmaS:         float64(radius) / 2,
		SigmaR:         sigmaR,
		NoiseReduction: noiseReduction,
	}
	out, err := Bilateral(g, p)
	if err != nil {
		return nil, nil, err
	}
	rep := &DenoiseReport{
		SigmaR:     sigmaR,
		Radius:     radius,
		SigmaS:     p.SigmaS,
		Derivation: append(s1, s2...),
	}
	ret, err := EdgeRetention(g, out)
	if err != nil {
		return nil, nil, err
	}
	rep.EdgeRetention = ret
	if flat != nil {
		if len(flat) != 4 {
			return nil, nil, fieldErr("Region", ErrBilateralRegion)
		}
		drop, err := VarianceDrop(g, out, flat[0], flat[1], flat[2], flat[3])
		if err != nil {
			return nil, nil, err
		}
		rep.VarianceDrop = drop
	}
	return out, rep, nil
}
