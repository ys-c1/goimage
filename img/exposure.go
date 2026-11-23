package img

import (
	"errors"
	"fmt"
	"math"
)

// 曝光补偿与拼接相关错误。所有图级错误都带图号（1 起）。
var (
	ErrOverlapEmpty   = errors.New("重叠区为空")
	ErrOverlapRange   = errors.New("重叠宽度越界")
	ErrHeightMismatch = errors.New("两图高度不一致")
	ErrEmptyImage     = errors.New("图为空")
	ErrNegativeGain   = errors.New("增益为负")
	ErrNoUsable       = errors.New("重叠区无可用样本")
)

// FigureError 把图号、细节与哨兵错误绑定，便于 errors.Is / errors.As 定位。
type FigureError struct {
	Figures string // "图 2" 或 "图 1|2"
	Detail  string
	Err     error
}

func (e *FigureError) Error() string {
	if e.Detail == "" {
		return fmt.Sprintf("%v：%s", e.Err, e.Figures)
	}
	return fmt.Sprintf("%v：%s（%s）", e.Err, e.Figures, e.Detail)
}

func (e *FigureError) Unwrap() error { return e.Err }

func figure(i int) string { return fmt.Sprintf("图 %d", i) }

// checkGray 校验单张图，idx 从 1 起，错误里报出图号。
func checkGray(g *Gray, idx int) error {
	if g == nil {
		return &FigureError{Figures: figure(idx), Detail: "指针为空", Err: ErrEmptyImage}
	}
	if g.W <= 0 || g.H <= 0 {
		return &FigureError{
			Figures: figure(idx),
			Detail:  fmt.Sprintf("画幅 %dx%d", g.W, g.H),
			Err:     ErrEmptyImage,
		}
	}
	if len(g.Pix) < g.W*g.H {
		return &FigureError{
			Figures: figure(idx),
			Detail:  fmt.Sprintf("像素缓冲长度 %d 小于画幅 %dx%d", len(g.Pix), g.W, g.H),
			Err:     ErrEmptyImage,
		}
	}
	return nil
}

// checkPair 校验一对图与重叠宽度。重叠宽度以较窄一图的宽度为上界。
func checkPair(a, b *Gray, overlap int) error {
	if err := checkGray(a, 1); err != nil {
		return err
	}
	if err := checkGray(b, 2); err != nil {
		return err
	}
	if a.H != b.H {
		return &FigureError{
			Figures: "图 1|2",
			Detail:  fmt.Sprintf("高度 %d 与 %d", a.H, b.H),
			Err:     ErrHeightMismatch,
		}
	}
	maxOverlap := a.W
	if b.W < maxOverlap {
		maxOverlap = b.W
	}
	if overlap < 0 || overlap > maxOverlap {
		return &FigureError{
			Figures: "图 1|2",
			Detail:  fmt.Sprintf("重叠宽度 %d 超过画幅 %d 或 %d", overlap, a.W, b.W),
			Err:     ErrOverlapRange,
		}
	}
	if overlap == 0 {
		return &FigureError{
			Figures: "图 1|2",
			Detail:  "重叠宽度 0",
			Err:     ErrOverlapEmpty,
		}
	}
	return nil
}

// StitchDirect 是原有的直通拼接：不做任何曝光补偿，图 1 全幅保留，
// 图 2 只取重叠区之后的独有部分。重叠区取图 1 原始像素，因此输出
// 与输入逐位一致。曝光补偿路径不会改变这里的行为。
func StitchDirect(a, b *Gray, overlap int) (*Gray, error) {
	if err := checkPair(a, b, overlap); err != nil {
		return nil, err
	}
	out, err := New(a.W+b.W-overlap, a.H)
	if err != nil {
		return nil, err
	}
	for y := 0; y < a.H; y++ {
		copy(out.Pix[y*out.W:y*out.W+a.W], a.Pix[y*a.W:y*a.W+a.W])
		for x := overlap; x < b.W; x++ {
			out.Pix[y*out.W+a.W+(x-overlap)] = b.Pix[y*b.W+x]
		}
	}
	return out, nil
}

// ExposureFit 是一对图的曝光拟合结果。
//
// 图 1 作为基准：Gain[0]=1、Bias[0]=0。图 2 的模型为
// I' = Gain[1]·I + Bias[1]。残余统计均按最终 8 位输出计算。
type ExposureFit struct {
	Base int // 恒为 0，表示以图 1 为基准

	Gain [2]float64
	Bias [2]float64

	// Usable / Saturated：重叠区中两侧都严格在 (0,255) 内的样本数，
	// 以及至少一侧落在 0 或 255 的样本数。拟合只用前者。
	Usable    int
	Saturated int

	// Undetermined 表示重叠区对比度不足，增益不可由数据唯一确定，
	// 此时由 Tikhonov 正则项把增益固定到 1。
	Undetermined bool
	// Lambda 为实际使用的正则强度。
	Lambda float64
	// RawGain 为不加正则项时的普通最小二乘增益（sxx=0 时记为 1）。
	RawGain float64

	// ResidualBefore / ResidualAfter：可用样本上的平均绝对亮度差，
	// 前为补偿前，后为补偿后（按最终 8 位、含四舍五入）。
	ResidualBefore float64
	ResidualAfter  float64
	// ResidualBeforeAll / ResidualAfterAll：把饱和样本也算进来的对照值。
	// 饱和像素在补偿前已丢失信息，任何增益/偏置都无法恢复。
	ResidualBeforeAll float64
	ResidualAfterAll  float64
	// ResidualAfterPlaced：重叠区按实际落位策略（饱和回退图 1）后的平均
	// 绝对亮度差，即最终输出与基准图在重叠区的残差。
	ResidualAfterPlaced float64

	// FallbackToBase：重叠区中图 2 已落在 0/255、图 1 仍严格在 (0,255)
	// 内的像素数；落位时这些像素改用图 1，避免输出残留饱和块。
	FallbackToBase int

	// NonOverlapClip：图 2 独有区补偿后落在 [0,255] 之外的像素数。
	NonOverlapClip int
	// NonOverlapMin / NonOverlapMax：图 2 独有区补偿后的取值范围（未钳制）。
	NonOverlapMin float64
	NonOverlapMax float64
}

// Apply 返回第 i 张图在曝光模型下的浮点补偿值。
func (f *ExposureFit) Apply(i int, v uint8) float64 {
	return f.Gain[i]*float64(v) + f.Bias[i]
}

// ApplyRound 返回第 i 张图补偿并钳制到 8 位后的值。
func (f *ExposureFit) ApplyRound(i int, v uint8) uint8 {
	return clampRound8(f.Apply(i, v))
}

// OrderReduced 判断重叠区残差是否至少缩小一个量级。
// 补偿前残差为零时，要求补偿后仍为零。
func (f *ExposureFit) OrderReduced() bool {
	if f.ResidualBefore == 0 {
		return f.ResidualAfter == 0
	}
	return f.ResidualAfter*10 <= f.ResidualBefore
}

func clampRound8(v float64) uint8 {
	if math.IsNaN(v) {
		return 0
	}
	r := math.Round(v)
	if r < 0 {
		return 0
	}
	if r > 255 {
		return 255
	}
	return uint8(r)
}

func interior(v uint8) bool { return v > 0 && v < 255 }

// FitExposure 用重叠区的亮度对应关系解出图 2 相对图 1 的增益与偏置。
//
// 对重叠区每一对同一场景点 (a_i, b_i) 建立
//
//	g1·a_i + c1 = g2·b_i + c2
//
// 四个未知量存在两个规范自由度：整体乘一个常数、整体加一个常数都不改变
// 等式。这里取图 1 为基准 (g1,c1)=(1,0)，于是化为对图 2 的
//
//	min Σ (g·b_i + c − a_i)² + λ·(g−1)²
//
// 拟合只用两侧都严格在 (0,255) 内的样本：0/255 在补偿前已被传感器截断，
// 不满足线性曝光关系，纳入拟合只会把增益拉偏。
func FitExposure(a, b *Gray, overlap int) (*ExposureFit, error) {
	if err := checkPair(a, b, overlap); err != nil {
		return nil, err
	}

	var (
		usable, saturated int
		sumA, sumB        float64
		sumBB, sumAB      float64
		beforeUsable      float64
	)
	for y := 0; y < a.H; y++ {
		for k := 0; k < overlap; k++ {
			av := a.Pix[y*a.W+(a.W-overlap+k)]
			bv := b.Pix[y*b.W+k]
			if interior(av) && interior(bv) {
				usable++
				x := float64(bv)
				yval := float64(av)
				sumA += yval
				sumB += x
				sumBB += x * x
				sumAB += x * yval
				beforeUsable += math.Abs(yval - x)
			} else {
				saturated++
			}
		}
	}
	if usable == 0 {
		return nil, &FigureError{
			Figures: "图 1|2",
			Detail:  fmt.Sprintf("重叠区 %d 个样本全部落在 0/255", overlap*a.H),
			Err:     ErrNoUsable,
		}
	}

	n := float64(usable)
	meanA := sumA / n
	meanB := sumB / n
	// 中心化和，等价于 Σ(b−b̄)² 与 Σ(b−b̄)(a−ā)。
	sxx := sumBB - n*meanB*meanB
	sxy := sumAB - n*meanB*meanA
	if sxx < 0 && sxx > -1e-9*math.Max(1, n) {
		sxx = 0
	}

	// Tikhonov 强度：λ = 1e-6·(Σb²+n)。理由见 docs/exposure.md。
	lambda := 1e-6 * (sumBB + n)

	rawGain := 1.0
	if sxx > 0 {
		rawGain = sxy / sxx
	}
	g := (sxy + lambda) / (sxx + lambda)
	c := meanA - g*meanB
	if g < 0 {
		return nil, &FigureError{
			Figures: figure(2),
			Detail:  fmt.Sprintf("增益 %.6f", g),
			Err:     ErrNegativeGain,
		}
	}

	f := &ExposureFit{
		Base:         0,
		Gain:         [2]float64{1, g},
		Bias:         [2]float64{0, c},
		Usable:       usable,
		Saturated:    saturated,
		Lambda:       lambda,
		RawGain:      rawGain,
		Undetermined: sxx <= 1e-9*math.Max(1, n),
	}

	// 残差（可用样本）+ 全样本对照。
	var beforeAll, afterAll, afterUsable, afterPlaced float64
	pairs := overlap * a.H
	for y := 0; y < a.H; y++ {
		for k := 0; k < overlap; k++ {
			av := a.Pix[y*a.W+(a.W-overlap+k)]
			bv := b.Pix[y*b.W+k]
			beforeAll += math.Abs(float64(av) - float64(bv))
			out := float64(f.ApplyRound(1, bv))
			diff := math.Abs(float64(av) - out)
			afterAll += diff
			if interior(av) && interior(bv) {
				afterUsable += diff
			}
			if !interior(bv) && interior(av) {
				f.FallbackToBase++
				// 落位时该点改用图 1，差值恰为 0。
				continue
			}
			afterPlaced += diff
		}
	}
	f.ResidualBefore = beforeUsable / n
	f.ResidualAfter = afterUsable / n
	f.ResidualBeforeAll = beforeAll / float64(pairs)
	f.ResidualAfterAll = afterAll / float64(pairs)
	f.ResidualAfterPlaced = afterPlaced / float64(pairs)

	// 不重叠区（图 2 独有列）的截断统计。
	first := true
	for y := 0; y < b.H; y++ {
		for x := overlap; x < b.W; x++ {
			v := f.Apply(1, b.Pix[y*b.W+x])
			if v < 0 || v > 255 {
				f.NonOverlapClip++
			}
			if first {
				f.NonOverlapMin, f.NonOverlapMax = v, v
				first = false
			} else {
				if v < f.NonOverlapMin {
					f.NonOverlapMin = v
				}
				if v > f.NonOverlapMax {
					f.NonOverlapMax = v
				}
			}
		}
	}
	return f, nil
}

// StitchExposure 按曝光拟合结果拼接两图：图 1 全幅保留，图 2 先做
// I'=g·I+c 再落位；重叠区优先用图 2 的补偿值，图 2 已饱和而图 1 未饱和
// 的像素改用图 1，避免最终输出残留饱和块。图 2 独有区如被钳制，
// 由 ExposureFit.NonOverlapClip 报出，不静默隐藏。
func StitchExposure(a, b *Gray, overlap int) (*Gray, *ExposureFit, error) {
	fit, err := FitExposure(a, b, overlap)
	if err != nil {
		return nil, nil, err
	}
	out, err := New(a.W+b.W-overlap, a.H)
	if err != nil {
		return nil, nil, err
	}
	// 图 1 原样落位（基准图）。
	for y := 0; y < a.H; y++ {
		copy(out.Pix[y*out.W:y*out.W+a.W], a.Pix[y*a.W:y*a.W+a.W])
	}
	// 图 2 独有区：补偿后落位，超出 8 位范围时钳制（同时已计数）。
	for y := 0; y < b.H; y++ {
		for x := overlap; x < b.W; x++ {
			out.Pix[y*out.W+a.W+(x-overlap)] = fit.ApplyRound(1, b.Pix[y*b.W+x])
		}
	}
	// 重叠区：图 2 补偿值；图 2 端点饱和而图 1 仍有效时改用图 1。
	for y := 0; y < a.H; y++ {
		for k := 0; k < overlap; k++ {
			av := a.Pix[y*a.W+(a.W-overlap+k)]
			bv := b.Pix[y*b.W+k]
			if !interior(bv) && interior(av) {
				out.Pix[y*out.W+(a.W-overlap+k)] = av
				continue
			}
			out.Pix[y*out.W+(a.W-overlap+k)] = fit.ApplyRound(1, bv)
		}
	}
	return out, fit, nil
}
