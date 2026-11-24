package img

import (
	"math"
	"time"
)

// 反卷积路径的常量。停止阈值由 img/deconv_test.go 的五幅用例锁定，
// 改动任一常数都应重跑该核验。
const (
	// 振铃上限：半级灰度。低于量化步长，因此 8 位输出里过冲严格为零。
	ringLimit = 0.5

	// Landweber 步长。核已归一化，故 ||K||₂ = max|H| = 1，步长取 1 仍收敛。
	landweberStep = 1.0

	// 振铃预算相对上一轮振铃的收缩比。每轮预算取上一轮实测振铃的一半，
	// 保证振铃幅度至少按 1/2 几何下降。
	budgetShrink = 0.5
)

// DeconvOptions 是 Deconvolve 的参数。
type DeconvOptions struct {
	// Kernel 为已知模糊核，必须可逆（频域符号无零点）。
	Kernel Kernel

	// Amount 为迭代初值的单次锐化量：初值 = g + Amount·(g − K·g)。
	// 取 0 表示直接从观测图起步。必须为有限非负数。
	Amount float64

	// MaxIter 为迭代上限，必须为正。
	MaxIter int

	// DisableRingBudget 关闭振铃预算约束，只保留非负与总和约束。
	// 零值（false）即正常路径：带预算、保证振铃递减且终值为零。
	// 置真只用于对照说明“只看残差为什么会振铃”，不作为交付路径。
	DisableRingBudget bool
}

// DeconvStep 记录一次迭代后的状态。
type DeconvStep struct {
	Iter       int     // 迭代序号，从 1 起
	Residual   float64 // 相对 L2 残差 ||Kx−g||₂/||g||₂
	RingAmp    float64 // 振铃幅度：越出观测图 3×3 局部值域的最大灰度级
	RingPixels int     // 振铃像素数：过冲超过半级的像素个数
	Budget     float64 // 本轮施加的振铃预算
}

// DeconvStats 是 Deconvolve 的核验数据。
type DeconvStats struct {
	ResidualStart   float64      // 单次锐化初值的残差
	ResidualBound   float64      // 残差下界：平凡解 x=g 的残差
	ResidualEnd     float64      // 最后一轮残差
	RingStart       float64      // 初值振铃幅度
	RingEnd         float64      // 最后一轮振铃幅度
	RingPixelsStart int          // 初值振铃像素数
	RingPixelsEnd   int          // 最后一轮振铃像素数
	Iterations      int          // 实际迭代次数
	Converged       bool         // 是否由联合下界停机
	Stop            string       // 停止原因：joint-bound 或 max-iter
	Steps           []DeconvStep // 逐轮轨迹（不含初值）
	Elapsed         time.Duration
}

// Deconvolve 按已知模糊核做迭代反卷积，返回反卷积图与逐轮统计。
//
// 每轮先做一次 Landweber 更新 x ← x + Kᵀ(g − Kx)，再依次施加：
//   - 非负约束；
//   - 总和约束：整体平移使 Σx = Σg；
//   - 振铃预算（默认开启）：把每个像素钳到观测图 3×3 局部值域
//     向外扩 Budget 的盒内；Budget 取上一轮实测振铃的一半。
//
// 停机由残差与振铃的共同下界给出：残差不超过平凡解 x=g 的残差，
// 且振铃幅度不超过半级。只看残差不够——Landweber 是半收敛的，
// 继续迭代会压低数据残差却放大噪声频段，振铃随之增大（见
// DisableRingBudget 对照与 docs/deconv.md）。
//
// 输入错误逐项拒绝并给出字段名：
//   - 核不可逆 → 字段 kernel（errors.Is 可匹配 ErrKernelSingular）；
//   - 图为空   → 字段 image；
//   - 迭代上限非正 → 字段 maxIter；
//   - 锐化量非法 → 字段 amount。
func Deconvolve(g *Gray, opt DeconvOptions) (*Gray, *DeconvStats, error) {
	if err := validateImage(g); err != nil {
		return nil, nil, err
	}
	if opt.MaxIter <= 0 {
		return nil, nil, &FieldError{Field: "maxIter", Err: ErrIterations}
	}
	if math.IsNaN(opt.Amount) || math.IsInf(opt.Amount, 0) || opt.Amount < 0 {
		return nil, nil, &FieldError{Field: "amount", Err: ErrAmount}
	}
	k, err := checkKernel(opt.Kernel)
	if err != nil {
		return nil, nil, err
	}
	started := time.Now()

	w, h, n := g.W, g.H, g.W*g.H
	obs := grayToFloat(g)
	var total float64
	for _, v := range obs {
		total += v
	}
	lo, hi := localRange3x3(obs, w, h)
	norm := math.Sqrt(dot(obs, obs))
	bound := relativeResidual(obs, obs, k, w, h, norm) // x=g 的残差

	// 初值：单次锐化（Amount=0 时即观测图本身）。与 SharpenUnsharp 同一公式。
	blurred := k.apply(obs, w, h, false)
	cur := make([]float64, n)
	for i := 0; i < n; i++ {
		cur[i] = clampFloat(obs[i]+opt.Amount*(obs[i]-blurred[i]), 0, 255)
	}
	stats := &DeconvStats{
		ResidualStart: relativeResidual(cur, obs, k, w, h, norm),
		ResidualBound: bound,
		Stop:          "max-iter",
	}
	stats.RingStart, stats.RingPixelsStart = ringStats(cur, lo, hi)

	useBudget := !opt.DisableRingBudget
	budget := 0.0
	if useBudget {
		budget = stats.RingStart * budgetShrink
	}
	lower := make([]float64, n)
	upper := make([]float64, n)
	for it := 1; it <= opt.MaxIter; it++ {
		resid := sub(obs, k.apply(cur, w, h, false)) // g − Kx
		grad := k.apply(resid, w, h, true)           // Kᵀ(g − Kx)
		for i := 0; i < n; i++ {
			cur[i] += landweberStep * grad[i]
		}
		for i := 0; i < n; i++ {
			lower[i] = 0
			upper[i] = 255
			if useBudget {
				lower[i] = clampFloat(lo[i]-budget, 0, 255)
				upper[i] = clampFloat(hi[i]+budget, 0, 255)
			}
		}
		cur = projectSumBox(cur, lower, upper, total)

		step := DeconvStep{Iter: it, Budget: budget}
		step.RingAmp, step.RingPixels = ringStats(cur, lo, hi)
		step.Residual = relativeResidual(cur, obs, k, w, h, norm)
		if step.RingAmp <= ringLimit && step.Residual <= bound && useBudget {
			// 停机轮再做一次预算为零的投影，把结果落回局部值域内部。
			// 这一步把“终值不超过半级”从观测结论变成构造保证——半级恰在
			// 8 位量化的进位边界上，停在 0.5 可能舍入成一级。投影会略微
			// 改变残差，故停机判定要对投影后的状态重新求值。
			for i := 0; i < n; i++ {
				lower[i] = clampFloat(lo[i], 0, 255)
				upper[i] = clampFloat(hi[i], 0, 255)
			}
			cur = projectSumBox(cur, lower, upper, total)
			step.RingAmp, step.RingPixels = ringStats(cur, lo, hi)
			step.Residual = relativeResidual(cur, obs, k, w, h, norm)
		}
		matched := step.RingAmp <= ringLimit && step.Residual <= bound
		stats.Steps = append(stats.Steps, step)

		if useBudget {
			budget = step.RingAmp * budgetShrink
			if budget < ringLimit {
				budget = 0
			}
		}
		if matched {
			stats.Converged = true
			stats.Stop = "joint-bound"
			break
		}
	}

	stats.Iterations = len(stats.Steps)
	stats.RingEnd, stats.RingPixelsEnd = ringStats(cur, lo, hi)
	stats.ResidualEnd = stats.ResidualStart
	if stats.Iterations > 0 {
		stats.ResidualEnd = stats.Steps[stats.Iterations-1].Residual
	}
	out, err := New(w, h)
	if err != nil {
		return nil, nil, err
	}
	for i := range out.Pix {
		out.Pix[i] = clampByte(cur[i])
	}
	stats.Elapsed = time.Since(started)
	return out, stats, nil
}

// localRange3x3 返回每个像素 3×3 邻域（重复边界，含自身）的最小与最大值。
// 邻域取自观测图 g，是独立于迭代结果的判据。
func localRange3x3(v []float64, w, h int) ([]float64, []float64) {
	lo := make([]float64, w*h)
	hi := make([]float64, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			mn, mx := math.Inf(1), math.Inf(-1)
			for dy := -1; dy <= 1; dy++ {
				yy := clampInt(y+dy, 0, h-1)
				for dx := -1; dx <= 1; dx++ {
					p := v[yy*w+clampInt(x+dx, 0, w-1)]
					if p < mn {
						mn = p
					}
					if p > mx {
						mx = p
					}
				}
			}
			lo[y*w+x], hi[y*w+x] = mn, mx
		}
	}
	return lo, hi
}

// ringStats 返回越出 [lo,hi] 的最大灰度级，以及过冲超过半级的像素数。
func ringStats(x, lo, hi []float64) (float64, int) {
	amp, count := 0.0, 0
	for i := range x {
		var e float64
		if x[i] < lo[i] {
			e = lo[i] - x[i]
		} else if x[i] > hi[i] {
			e = x[i] - hi[i]
		}
		if e > amp {
			amp = e
		}
		if e > ringLimit {
			count++
		}
	}
	return amp, count
}

func relativeResidual(x, obs []float64, k Kernel, w, h int, obsNorm float64) float64 {
	if obsNorm == 0 {
		return 0
	}
	d := sub(k.apply(x, w, h, false), obs)
	return math.Sqrt(dot(d, d)) / obsNorm
}

// projectSumBox 求 min ||x−y||² s.t. Σx=S 且 L≤x≤U。
// KKT 解形如 x=clip(y+t,L,U)，t 由 Σclip(y+t,L,U)=S 唯一确定，
// 用二分求出（该函数关于 t 单调不减）。
func projectSumBox(y, L, U []float64, S float64) []float64 {
	f := func(t float64) float64 {
		var s float64
		for i := range y {
			s += clampFloat(y[i]+t, L[i], U[i])
		}
		return s - S
	}
	a, b := -1e9, 1e9
	for i := 0; i < 200; i++ {
		m := (a + b) / 2
		if f(m) < 0 {
			a = m
		} else {
			b = m
		}
	}
	t := (a + b) / 2
	out := make([]float64, len(y))
	for i := range y {
		out[i] = clampFloat(y[i]+t, L[i], U[i])
	}
	return out
}

func sub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// SharpenUnsharp 是原有的单次锐化基线，行为保持不变：
// x = g + amount·(g − K·g)，钳到 [0,255] 后四舍五入。
// amount 必须为有限非负数，错误字段名为 amount。
func SharpenUnsharp(g *Gray, k Kernel, amount float64) (*Gray, error) {
	if err := validateImage(g); err != nil {
		return nil, err
	}
	if math.IsNaN(amount) || math.IsInf(amount, 0) || amount < 0 {
		return nil, &FieldError{Field: "amount", Err: ErrAmount}
	}
	// 单次锐化只要求结构合法的核；可逆性是反卷积路径的专属条件，
	// 不施加到基线上（方框核做非锐化掩膜是常见做法）。
	nk, err := normalizeKernel(k.W, k.Size)
	if err != nil {
		return nil, err
	}
	blurred := nk.apply(grayToFloat(g), g.W, g.H, false)
	out, err := New(g.W, g.H)
	if err != nil {
		return nil, err
	}
	for i := range out.Pix {
		out.Pix[i] = clampByte(float64(g.Pix[i]) + amount*(float64(g.Pix[i])-blurred[i]))
	}
	return out, nil
}

// MaxGradient 返回相邻像素灰度差的最大绝对值，作为锐度的粗略指标。
func MaxGradient(g *Gray) int {
	if g == nil || g.W <= 0 || g.H <= 0 {
		return 0
	}
	best := 0.0
	for y := 0; y < g.H; y++ {
		for x := 1; x < g.W; x++ {
			d := math.Abs(float64(g.Pix[y*g.W+x]) - float64(g.Pix[y*g.W+x-1]))
			if d > best {
				best = d
			}
		}
	}
	for y := 1; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			d := math.Abs(float64(g.Pix[y*g.W+x]) - float64(g.Pix[(y-1)*g.W+x]))
			if d > best {
				best = d
			}
		}
	}
	return int(best)
}
