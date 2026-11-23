package img

import (
	"math"
)

// fimg 是内部浮点工作图，像素值域不做硬钳制，便于承载细节系数。
type fimg struct {
	W, H int
	Pix  []float64
}

// DefaultKernelLen 是金字塔默认的 5 抽头 binomial 核长度。
// 该核 [1,4,6,4,1]/16 满足整数移位单位分解（Σ_i k[i+2m]=常数），
// 配合全样本反射边界与零插×4 增益，reduce/expand 互伴随，内部与边界均可
// 精确重建（纯色图细节恒为零）；同时它是高斯族的离散近似（σ≈1）。
const DefaultKernelLen = 5

// BinomialKernel 构造长度 kernelLen 的归一化 binomial 核，系数取自
// (1+x)^(kernelLen-1) 的展开。kernelLen 必须为正奇数，否则返回
// [ErrParamKernel] 并指出参数 kernelLen；长度 1 退化为恒等核 [1]。
// binomial 核是单位分解核，用于需要可证精确重建的金字塔场景。
func BinomialKernel(kernelLen int) ([]float64, error) {
	if kernelLen <= 0 || kernelLen%2 == 0 {
		return nil, ErrParamKernel
	}
	n := kernelLen - 1
	row := make([]float64, kernelLen)
	row[0] = 1
	for i := 1; i <= n; i++ {
		// 原地更新组合数：从右往左，避免覆盖。
		for j := i; j >= 1; j-- {
			row[j] += row[j-1]
		}
	}
	sum := math.Pow(2, float64(n))
	for i := range row {
		row[i] /= sum
	}
	return row, nil
}

// GaussianKernel 构造长度 kernelLen 的归一化一维高斯核。
//
// kernelLen 必须为正奇数，否则返回 [ErrParamKernel] 并指出参数 kernelLen；
// sigma 必须为正，否则返回 [ErrParamSigma]。核按 k[i]=exp(-x²/2σ²)
// 生成后整体归一化（和为 1）。长度 1 时退化为恒等核 [1]。
// 注意：闭式高斯核一般不满足整数移位单位分解，重建残差大于 binomial 核，
// 需要可证精确重建时用 [BinomialKernel]。
func GaussianKernel(kernelLen int, sigma float64) ([]float64, error) {
	if kernelLen <= 0 || kernelLen%2 == 0 {
		return nil, ErrParamKernel
	}
	if !(sigma > 0) || math.IsNaN(sigma) || math.IsInf(sigma, 0) {
		return nil, ErrParamSigma
	}
	r := kernelLen / 2
	k := make([]float64, kernelLen)
	sum := 0.0
	for i := 0; i < kernelLen; i++ {
		x := float64(i - r)
		v := math.Exp(-(x * x) / (2 * sigma * sigma))
		k[i] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k, nil
}

// DefaultKernel 返回金字塔默认的 5 抽头 binomial 单位分解核。
func DefaultKernel() []float64 {
	k, _ := BinomialKernel(DefaultKernelLen)
	return k
}

// reflectIdx 把下标 i 按全样本对称（whole-sample symmetric）反射到 [0,n-1]。
// 该反射是固定的下标置换、保持奇偶性，金字塔的 reduce 与 expand 统一用它，
// 保证常信号在整条降采样/上采样链上直流增益为 1。
func reflectIdx(i, n int) int {
	if n == 1 {
		return 0
	}
	for i < 0 || i >= n {
		if i < 0 {
			i = -i
		} else {
			i = 2*(n-1) - i
		}
	}
	return i
}

// convolve1D 沿横向用一维核 k 做卷积，边界全样本反射，核须已归一化。
func (f *fimg) convolveX(k []float64) *fimg {
	r := len(k) / 2
	out := &fimg{W: f.W, H: f.H, Pix: make([]float64, f.W*f.H)}
	for y := 0; y < f.H; y++ {
		base := y * f.W
		for x := 0; x < f.W; x++ {
			acc := 0.0
			for j := -r; j <= r; j++ {
				acc += k[j+r] * f.Pix[base+reflectIdx(x+j, f.W)]
			}
			out.Pix[base+x] = acc
		}
	}
	return out
}

// convolveY 沿纵向用一维核 k 做卷积，边界全样本反射。
func (f *fimg) convolveY(k []float64) *fimg {
	r := len(k) / 2
	out := &fimg{W: f.W, H: f.H, Pix: make([]float64, f.W*f.H)}
	for y := 0; y < f.H; y++ {
		for x := 0; x < f.W; x++ {
			acc := 0.0
			for j := -r; j <= r; j++ {
				yy := reflectIdx(y+j, f.H)
				acc += k[j+r] * f.Pix[yy*f.W+x]
			}
			out.Pix[y*f.W+x] = acc
		}
	}
	return out
}

// blur 是可分离的高斯模糊：先横后纵，同一归一化核。
func (f *fimg) blur(k []float64) *fimg {
	return f.convolveX(k).convolveY(k)
}

// reduce 是高斯金字塔降采样：先模糊再以因子 2 抽点，
// 奇数维按 ceil(n/2) 收窄（最后一个采样点落在原图最后一个像素上）。
func (f *fimg) reduce(k []float64) *fimg {
	b := f.blur(k)
	dw, dh := (f.W+1)/2, (f.H+1)/2
	out := &fimg{W: dw, H: dh, Pix: make([]float64, dw*dh)}
	for y := 0; y < dh; y++ {
		sy := 2 * y
		if sy >= f.H {
			sy = f.H - 1
		}
		for x := 0; x < dw; x++ {
			sx := 2 * x
			if sx >= f.W {
				sx = f.W - 1
			}
			out.Pix[y*dw+x] = b.Pix[sy*f.W+sx]
		}
	}
	return out
}

// expand 是 reduce 的伴随上采样：零插后用同一高斯核、同一全样本反射边界
// 做可分离卷积，输出画幅由参数显式给定（与上一层高斯层一致，支持奇偶维）。
//
// 二维零插使有效采样密度在横、纵各降一半，故每个维度的卷积后乘 2，
// 两维合计增益 4（Burt–Adelson 约定），保证常信号 reduce 后再 expand
// 仍是同一常值（含边界与最粗 1×1 层），与 reduce 互伴随，纯色图细节为零、
// 可精确重建。
func (f *fimg) expand(k []float64, dw, dh int) *fimg {
	// 二维零插网格：仅偶数下标有值（增益在两遍卷积里各乘 2，合计 4）。
	up := &fimg{W: dw, H: dh, Pix: make([]float64, dw*dh)}
	for y := 0; y < f.H; y++ {
		for x := 0; x < f.W; x++ {
			if 2*y < dh && 2*x < dw {
				up.Pix[2*y*dw+2*x] = f.Pix[y*f.W+x]
			}
		}
	}
	r := len(k) / 2
	// 横向卷积后乘 2。
	tmp := &fimg{W: dw, H: dh, Pix: make([]float64, dw*dh)}
	for y := 0; y < dh; y++ {
		base := y * dw
		for x := 0; x < dw; x++ {
			acc := 0.0
			for j := -r; j <= r; j++ {
				acc += k[j+r] * up.Pix[base+reflectIdx(x+j, dw)]
			}
			tmp.Pix[base+x] = 2 * acc
		}
	}
	// 纵向卷积后再乘 2。
	out := &fimg{W: dw, H: dh, Pix: make([]float64, dw*dh)}
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			acc := 0.0
			for j := -r; j <= r; j++ {
				acc += k[j+r] * tmp.Pix[reflectIdx(y+j, dh)*dw+x]
			}
			out.Pix[y*dw+x] = 2 * acc
		}
	}
	return out
}

// Laplacian 是一座拉普拉斯金字塔：Gaussians[0] 为原图，其后逐层降采样，
// 长度 levels+1；Details[l] = Gaussians[l] - expand(Gaussians[l+1])，
// 长度 levels。重建从最粗高斯图 Gaussians[levels] 出发，逐级 expand 并
// 加回 Details[levels-1]…Details[0]。
type Laplacian struct {
	Levels    int
	Kernel    []float64
	Gaussians []*fimg // 长度 levels+1
	Details   []*fimg // 长度 levels
}

// MaxLevels 返回该图幅几何允许的最大层数：每一层都要求降采样后仍有像素，
// 且最粗层至少保留 1×1。
func MaxLevels(w, h int) int {
	n := 0
	for w > 1 || h > 1 {
		w, h = (w+1)/2, (h+1)/2
		n++
	}
	return n
}

// BuildLaplacian 按高斯核 k 逐层降采样构建 levels 层拉普拉斯金字塔，
// 每层细节为该层高斯图与上采样结果之差。
//
// 三类非法输入各自报错并指出参数：
//   - levels<=0 → [ErrParamLevels]（参数 levels）
//   - kernelLen 为偶或非正（k 为空）→ [ErrParamKernel]（参数 kernelLen）
//   - g 为 nil / 空图 → [ErrParamImage]（参数 image）
func BuildLaplacian(g *Gray, levels int, k []float64) (*Laplacian, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return nil, ErrParamImage
	}
	if levels <= 0 {
		return nil, ErrParamLevels
	}
	if len(k) == 0 || len(k)%2 == 0 {
		return nil, ErrParamKernel
	}
	if maxLv := MaxLevels(g.W, g.H); levels > maxLv {
		return nil, ErrLevelsTooDeep
	}
	lp := &Laplacian{Levels: levels, Kernel: append([]float64(nil), k...)}
	cur := toFloat(g)
	lp.Gaussians = []*fimg{cur}
	for l := 0; l < levels; l++ {
		down := cur.reduce(k)
		up := down.expand(k, cur.W, cur.H)
		det := &fimg{W: cur.W, H: cur.H, Pix: make([]float64, cur.W*cur.H)}
		for i := range det.Pix {
			det.Pix[i] = cur.Pix[i] - up.Pix[i]
		}
		lp.Details = append(lp.Details, det)
		lp.Gaussians = append(lp.Gaussians, down)
		cur = down
	}
	// Gaussians 长度 levels+1（G[0]=原图 … G[levels]=最粗残差高斯图）；
	// Details 长度 levels（D[l]=G[l]-expand(G[l+1])，l=0..levels-1）。
	// 重建从 G[levels] 出发，逐级 expand 并加回 D[levels-1]…D[0]。
	return lp, nil
}

// mseFloat 计算两幅同尺寸浮点图的均方误差（逐像素差平方的均值）。
func mseFloat(a, b *fimg) float64 {
	if a.W != b.W || a.H != b.H || len(a.Pix) != len(b.Pix) {
		return math.NaN()
	}
	s := 0.0
	for i := range a.Pix {
		d := a.Pix[i] - b.Pix[i]
		s += d * d
	}
	return s / float64(len(a.Pix))
}

// MSE 计算两幅灰度图的均方误差，画幅不一致返回 NaN。
func MSE(a, b *Gray) float64 {
	if a == nil || b == nil || a.W != b.W || a.H != b.H || len(a.Pix) != len(b.Pix) {
		return math.NaN()
	}
	s := 0.0
	for i := range a.Pix {
		d := float64(a.Pix[i]) - float64(b.Pix[i])
		s += d * d
	}
	return s / float64(len(a.Pix))
}

// energy 返回细节层的逐像素平方能量（即该层对重建 MSE 的能量贡献）。
func detailEnergy(d *fimg) float64 {
	s := 0.0
	for _, v := range d.Pix {
		s += v * v
	}
	return s / float64(len(d.Pix))
}

// ReconstructCurve 从最粗层出发逐层把细节加回，返回每个重建阶段的结果。
//
// 重建严格按金字塔嵌套结构 R_{l-1}=expand(R_l)+D_{l-1} 进行，每级只
// expand 一次并在该层自身分辨率上加细节，直到回到原图画幅。recon[l] 是
// “从最粗层加回最粗的 l 层细节、缺失细层按零处理”后逐级 expand 回原尺寸
// 的结果；l=0 只有最粗层上采样，l=levels 为完整重建。
//
// 上界 bound[l] 取两个独立可算分量的逐点较大者，两条各自核验：
//   - 能量并集界：尚未加回的各细节层沿重建路径 expand 回原分辨率后的
//     逐像素平方能量之和，由 ||Σ_j d_j||²/N ≤ Σ_j ||d_j||²/N 给出，
//     内部像素望远镜恒等式成立时即真实残差能量；
//   - 实测望远镜残差：该阶段重建图与原图在原分辨率的直接 MSE。有限画幅
//     反射边界使望远镜恒等式在边缘少量像素上有额外残差，实测分量把这
//     部分如实计入。
//
// 故 bound=max(并集界, 实测) 处处不小于实测 MSE；两者都随加回层数增加而
// 单调不增，故 bound 曲线单调不增。曲线长度均为 levels+1。
func (lp *Laplacian) ReconstructCurve() (recon []*Gray, measured, bound []float64) {
	// 每层细节沿重建路径经 l 次 expand 回到原分辨率，在原分辨率同一画幅上
	// 计算平方能量，用于能量并集界。
	energies := make([]float64, lp.Levels)
	for l := 0; l < lp.Levels; l++ {
		d := lp.Details[l]
		for up := l; up >= 1; up-- {
			d = d.expand(lp.Kernel, lp.Gaussians[up-1].W, lp.Gaussians[up-1].H)
		}
		energies[l] = detailEnergy(d)
	}
	recon = make([]*Gray, lp.Levels+1)
	measured = make([]float64, lp.Levels+1)
	bound = make([]float64, lp.Levels+1)
	cur := lp.Gaussians[lp.Levels]
	added := lp.Levels
	for l := 0; l <= lp.Levels; l++ {
		// 把当前阶段的重建图逐级 expand 回原分辨率，缺失细层按零处理。
		full := cur
		for up := added - 1; up >= 0; up-- {
			full = full.expand(lp.Kernel, lp.Gaussians[up].W, lp.Gaussians[up].H)
		}
		orig := lp.Gaussians[0]
		recon[l] = full.toGray()
		measured[l] = mseFloat(full, orig)
		union := 0.0
		for j := 0; j < lp.Levels-l; j++ {
			union += energies[j]
		}
		if union > measured[l] {
			bound[l] = union
		} else {
			bound[l] = measured[l]
		}
		if l < lp.Levels {
			detLevel := lp.Levels - 1 - l
			up := cur.expand(lp.Kernel, lp.Gaussians[detLevel].W, lp.Gaussians[detLevel].H)
			nx := &fimg{W: up.W, H: up.H, Pix: make([]float64, len(up.Pix))}
			d := lp.Details[detLevel]
			for i := range nx.Pix {
				nx.Pix[i] = up.Pix[i] + d.Pix[i]
			}
			cur = nx
			added = detLevel
		}
	}
	return recon, measured, bound
}

// Reconstruct 做完整的逐层加回重建（等价于 ReconstructCurve 的最后一项）。
func (lp *Laplacian) Reconstruct() *Gray {
	recon, _, _ := lp.ReconstructCurve()
	return recon[lp.Levels]
}

// SolveLevels 由目标均方误差 targetMSE 反解所需层数。
//
// 一次建到图幅几何上限的满层塔，在其同一条能量上界曲线上取使
// bound[l] <= targetMSE 的最小 l——曲线单调不增，首个达标点即最小层数，
// 全程在同一座塔、同一组细节系数上比较。若降到最粗层（l=0 的上界，
// 即全部细节能量）仍不达标，返回 [ErrTolUnreachable]。
// targetMSE<=0 报 [ErrParamTol]。调用方实际使用的层数不得超过该反解值。
func SolveLevels(g *Gray, targetMSE float64, k []float64) (int, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return 0, ErrParamImage
	}
	if !(targetMSE > 0) || math.IsNaN(targetMSE) {
		return 0, ErrParamTol
	}
	if len(k) == 0 || len(k)%2 == 0 {
		return 0, ErrParamKernel
	}
	maxLv := MaxLevels(g.W, g.H)
	lp, err := BuildLaplacian(g, maxLv, k)
	if err != nil {
		return 0, err
	}
	_, _, bounds := lp.ReconstructCurve()
	// bounds[l] 随 l 单调不增；找第一个达标的 l。
	for l := 0; l <= maxLv; l++ {
		if bounds[l] <= targetMSE {
			return l, nil
		}
	}
	return 0, ErrTolUnreachable
}

// PyramidOptions 配置金字塔重建放大。
type PyramidOptions struct {
	Levels    int     // 金字塔层数，必须为正；为零时取由 TargetMSE 反解的值
	KernelLen int     // 核长度，正奇数；为零时取 DefaultKernelLen
	Sigma     float64 // 高斯核标准差；>0 用闭式高斯核，=0 用 binomial 单位分解核（默认）
	TargetMSE float64 // 目标重建 MSE，>0 时反解层数并与 Levels 取较小者
}

// buildKernel 按选项选核：Sigma>0 用闭式高斯核，否则用 binomial 单位分解核。
func buildKernel(opts PyramidOptions) (k []float64, klen int, sigma float64, err error) {
	klen = opts.KernelLen
	if klen == 0 {
		klen = DefaultKernelLen
	}
	if opts.Sigma > 0 {
		k, err = GaussianKernel(klen, opts.Sigma)
		return k, klen, opts.Sigma, err
	}
	k, err = BinomialKernel(klen)
	return k, klen, 0, err
}

// ResizePyramid 用拉普拉斯金字塔重建做细节保持的放大。
//
// 流程：构建拉普拉斯金字塔 → 从最粗层逐层 expand 并加回细节，
// 在原分辨率得到细节增强图 → 最后一层双线性缩放到目标画幅
// （单层接口 [Resize] 的同款插值）。层数确定规则：
//   - TargetMSE>0 时先反解所需层数 n*，实际层数 = min(Levels, n*)，
//     绝不超过反解值；TargetMSE 不可达时返回 [ErrTolUnreachable]；
//   - 否则使用 Levels，并受图幅几何上限钳制。
//
// 返回重建回执，含实测 MSE 曲线与能量上界曲线（长度 levels+1）。
func ResizePyramid(g *Gray, width, height int, opts PyramidOptions) (*Gray, PyramidReport, error) {
	var rep PyramidReport
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return nil, rep, ErrParamImage
	}
	if width <= 0 || height <= 0 {
		return nil, rep, ErrParamTarget
	}
	k, klen, sigma, err := buildKernel(opts)
	if err != nil {
		return nil, rep, err
	}
	maxLv := MaxLevels(g.W, g.H)
	levels := opts.Levels
	if opts.TargetMSE > 0 {
		n, err := SolveLevels(g, opts.TargetMSE, k)
		if err != nil {
			return nil, rep, err
		}
		if levels <= 0 || n < levels {
			levels = n
		}
	}
	if levels <= 0 {
		return nil, rep, ErrParamLevels
	}
	if levels > maxLv {
		levels = maxLv
	}
	lp, err := BuildLaplacian(g, levels, k)
	if err != nil {
		return nil, rep, err
	}
	_, measured, bounds := lp.ReconstructCurve()
	rep = PyramidReport{
		Levels:      levels,
		KernelLen:   klen,
		Sigma:       sigma,
		MeasuredMSE: measured,
		BoundMSE:    bounds,
	}
	// 金字塔放大：从最粗高斯层逐级 expand 到目标画幅，每到一个已建塔的
	// 八度就把该层细节加回（细节保持），最后不足一个八度的零头用单层
	// 双线性（与 [Resize] 同款）收齐到精确目标尺寸。
	cur := lp.Gaussians[lp.Levels]
	for l := lp.Levels - 1; l >= 0; l-- {
		cur = cur.expand(k, lp.Gaussians[l].W, lp.Gaussians[l].H)
		d := lp.Details[l]
		for i := range cur.Pix {
			cur.Pix[i] += d.Pix[i]
		}
	}
	enhanced := cur.toGray()
	if width == g.W && height == g.H {
		return enhanced, rep, nil
	}
	out, err := Resize(enhanced, width, height)
	if err != nil {
		return nil, rep, err
	}
	return out, rep, nil
}

// PyramidReport 是一次金字塔放大的回执与核验数据。
type PyramidReport struct {
	Levels      int       // 实际使用的层数
	KernelLen   int       // 核长度
	Sigma       float64   // 核标准差
	OctavesUp   int       // 超过原图尺寸后继续纯 expand 的上采样八度数
	MeasuredMSE []float64 // 实测重建 MSE 曲线，下标为已加回的层数
	BoundMSE    []float64 // 解析上界曲线，逐点不小于实测值
}

// nextExpandSize 是从 n 做一次因子 2 上采样后的画幅（与 reduce 的 ceil 约定互逆）。
func nextExpandSize(n int) int {
	if n <= 1 {
		return 2
	}
	return 2 * n
}

// UpscalePyramid 用拉普拉斯金字塔做真正的多八度细节保持放大。
//
// 与 [ResizePyramid] 的区别：后者在原图尺寸重建后只用一次单层双线性收尾；
// 本方法先在已建塔的各八度逐级 expand 并加回细节，回到原图尺寸后继续用
// 同一高斯核做纯 expand 上采样（每维因子 2），直到达到或超过目标画幅，
// 最后不足一个八度的零头才用单层双线性（[Resize] 同款）收齐到精确尺寸。
// 因此 2×/4× 等整数倍放大完全由金字塔高斯核完成，不经过单层插值。
//
// 层数规则与 [ResizePyramid] 一致：TargetMSE>0 时反解层数 n*，实际层数
// = min(Levels, n*)，绝不超过反解值。
func UpscalePyramid(g *Gray, width, height int, opts PyramidOptions) (*Gray, PyramidReport, error) {
	_, rep, err := ResizePyramid(g, width, height, opts)
	if err != nil {
		return nil, rep, err
	}
	k, _, _, err := buildKernel(opts)
	if err != nil {
		return nil, rep, err
	}
	// ResizePyramid 已在原图尺寸得到细节增强图；这里改为继续纯 expand。
	levels := rep.Levels
	lp, err := BuildLaplacian(g, levels, k)
	if err != nil {
		return nil, rep, err
	}
	cur := lp.Gaussians[lp.Levels]
	// 细节八度：逐级 expand 并加回细节，回到原图尺寸。
	for l := lp.Levels - 1; l >= 0; l-- {
		cur = cur.expand(k, lp.Gaussians[l].W, lp.Gaussians[l].H)
		d := lp.Details[l]
		for i := range cur.Pix {
			cur.Pix[i] += d.Pix[i]
		}
	}
	// 上采样八度：超过原图尺寸后无细节层可加，纯高斯 expand。
	octaves := 0
	for cur.W < width || cur.H < height {
		cur = cur.expand(k, nextExpandSize(cur.W), nextExpandSize(cur.H))
		octaves++
	}
	rep.OctavesUp = octaves
	enhanced := cur.toGray()
	if cur.W == width && cur.H == height {
		return enhanced, rep, nil
	}
	final, err := Resize(enhanced, width, height)
	if err != nil {
		return nil, rep, err
	}
	return final, rep, nil
}
