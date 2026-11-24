package img

import (
	"errors"
	"fmt"
	"math"
)

// 反卷积路径的输入校验错误。FieldError 的 Error 以非法字段名开头，
// 便于定位到底哪一处输入越界或为空。
var (
	ErrKernel         = errors.New("模糊核非法")
	ErrKernelSingular = errors.New("模糊核不可逆：频域符号存在零点")
	ErrEmptyImage     = errors.New("图为空")
	ErrIterations     = errors.New("迭代上限必须为正")
	ErrAmount         = errors.New("锐化量必须为非负有限值")
)

// FieldError 在错误信息前带上非法字段名。
type FieldError struct {
	Field string
	Err   error
}

func (e *FieldError) Error() string     { return e.Field + ": " + e.Err.Error() }
func (e *FieldError) Unwrap() error     { return e.Err }
func (e *FieldError) FieldName() string { return e.Field }

// 核可逆性判据常数：归一化模糊核频率响应 |H| 的最小值与最大值之比。
// 低于 1e-6 意味着某个频段被抹掉到百万分之一以下，反解会把该频段连同
// 噪声放大百万倍以上，视为不可逆。3x3 与 5x5 方框核在 (π/2, π/2) 处有
// 零点；3x3 高斯的一维符号在 σ > 1/√(2·ln2) ≈ 0.8493 时变号（Nyquist 处
// 为 2e^(−1/2σ²) − 1）。这些核都会被拒。
const (
	kernelSymbolFloor = 1e-6
	kernelSymbolGrid  = 64
	kernelSymbolSeeds = 16
)

// Kernel 是一个方形二维模糊核，行主序存放。
// 由 NewKernel / GaussianKernel 构造时元素已归一化为和为一；
// 直接用结构体字面量构造的核会在使用时再次校验并归一化。
type Kernel struct {
	Size int
	W    []float64
}

// NewKernel 校验并归一化一个 size x size 的二维核。
//
// 拒绝：size 非正或为偶数（无中心，无法对齐）、元素个数不符、
// 存在 NaN/Inf 或负权重、权重和不为正、以及频域不可逆的核。
// 非法元素报 kernel[i]，结构性错误报 kernel 或 kernel.size。
func NewKernel(w []float64, size int) (Kernel, error) {
	k, err := normalizeKernel(w, size)
	if err != nil {
		return Kernel{}, err
	}
	if r := k.symbolRatio(); r < kernelSymbolFloor {
		return Kernel{}, &FieldError{Field: "kernel", Err: ErrKernelSingular}
	}
	return k, nil
}

// GaussianKernel 返回归一化的 size x size 高斯核。
// size 必须为奇数，sigma 必须为有限正数；error 的字段名为 size 或 sigma。
func GaussianKernel(size int, sigma float64) (Kernel, error) {
	if size <= 0 || size%2 == 0 {
		return Kernel{}, &FieldError{Field: "size", Err: ErrKernel}
	}
	if math.IsNaN(sigma) || math.IsInf(sigma, 0) || sigma <= 0 {
		return Kernel{}, &FieldError{Field: "sigma", Err: ErrKernel}
	}
	r := size / 2
	w := make([]float64, size*size)
	for y := -r; y <= r; y++ {
		for x := -r; x <= r; x++ {
			w[(y+r)*size+(x+r)] = math.Exp(-float64(x*x+y*y) / (2 * sigma * sigma))
		}
	}
	return NewKernel(w, size)
}

// normalizeKernel 做结构性校验并按和归一化，不检查频域可逆性。
func normalizeKernel(w []float64, size int) (Kernel, error) {
	if size <= 0 || size%2 == 0 {
		return Kernel{}, &FieldError{Field: "kernel.size", Err: ErrKernel}
	}
	if len(w) != size*size {
		return Kernel{}, &FieldError{Field: "kernel", Err: ErrKernel}
	}
	var sum float64
	for i, v := range w {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return Kernel{}, &FieldError{Field: fmt.Sprintf("kernel[%d]", i), Err: ErrKernel}
		}
		sum += v
	}
	if !(sum > 0) {
		return Kernel{}, &FieldError{Field: "kernel", Err: ErrKernel}
	}
	out := make([]float64, len(w))
	for i, v := range w {
		out[i] = v / sum
	}
	return Kernel{Size: size, W: out}, nil
}

// checkKernel 校验任意来源的 Kernel（含结构体字面量），返回归一化副本。
func checkKernel(k Kernel) (Kernel, error) {
	nk, err := normalizeKernel(k.W, k.Size)
	if err != nil {
		return Kernel{}, err
	}
	if r := nk.symbolRatio(); r < kernelSymbolFloor {
		return Kernel{}, &FieldError{Field: "kernel", Err: ErrKernelSingular}
	}
	return nk, nil
}

// symbolRatio 返回核在频域 |H(ω)| 的最小/最大比，用于判断可逆性。
//
// 只在网格点上取样会漏掉落在格点之间的零点：3x3 方框核的零点在
// (π/2, π/2)，64 点网格上 |H| 仍为 3.6e-4，足以骗过阈值。因此先做
// 全网格扫描，再对扫描出的若干低谷做局部极小化（二维黄金分割），
// 取扫描值与细化值的较小者。
func (k Kernel) symbolRatio() float64 {
	if k.Size <= 0 || k.Size%2 == 0 || len(k.W) != k.Size*k.Size {
		return 0
	}
	dips := make([]symbolDip, 0, kernelSymbolSeeds)
	hi := 0.0
	for by := 0; by < kernelSymbolGrid; by++ {
		wy := 2 * math.Pi * float64(by) / kernelSymbolGrid
		for bx := 0; bx < kernelSymbolGrid; bx++ {
			wx := 2 * math.Pi * float64(bx) / kernelSymbolGrid
			a := k.symbolAt(wx, wy)
			if a > hi {
				hi = a
			}
			dips = insertDip(dips, symbolDip{wx, wy, a})
		}
	}
	if hi == 0 {
		return 0
	}
	// 网格扫描只保留最低的 kernelSymbolSeeds 个格点作种子，再做局部细化。
	// 周期为 2π，故种子可落在任意相位。
	lo := math.Inf(1)
	for i := 0; i < len(dips); i++ {
		a := k.refineSymbol(dips[i].wx, dips[i].wy, dips[i].amp)
		if a < lo {
			lo = a
		}
	}
	return lo / hi
}

// symbolAt 返回归一化频率 (wx, wy) 处频率响应的模。
func (k Kernel) symbolAt(wx, wy float64) float64 {
	r := k.Size / 2
	var re, im float64
	for ky := 0; ky < k.Size; ky++ {
		for kx := 0; kx < k.Size; kx++ {
			phase := wx*float64(kx-r) + wy*float64(ky-r)
			w := k.W[ky*k.Size+kx]
			re += w * math.Cos(phase)
			im -= w * math.Sin(phase)
		}
	}
	return math.Hypot(re, im)
}

// refineSymbol 从 (wx, wy) 出发做二维黄金分割，返回邻域内的极小 |H|。
// 频率响应关于 ω 以 2π 为周期，故搜索区间按模 2π 归一到 [-π, π]。
func (k Kernel) refineSymbol(wx, wy, amp float64) float64 {
	const phi = 0.6180339887498949
	half := math.Pi / float64(kernelSymbolGrid) * 4 // 种子间隔的 4 倍
	golden := func(f func(float64) float64, a, b float64) (float64, float64) {
		c := b - phi*(b-a)
		d := a + phi*(b-a)
		fc, fd := f(c), f(d)
		for i := 0; i < 60; i++ {
			if fc < fd {
				b, d, fd = d, c, fc
				c = b - phi*(b-a)
				fc = f(c)
			} else {
				a, c, fc = c, d, fd
				d = a + phi*(b-a)
				fd = f(d)
			}
		}
		if fc < fd {
			return c, fc
		}
		return d, fd
	}
	best := amp
	for i := 0; i < 3; i++ { // 坐标轮换三轮；每次都在当前最优点重开区间
		var v float64
		wx, v = golden(func(x float64) float64 { return k.symbolAt(x, wy) }, wx-half, wx+half)
		if v < best {
			best = v
		}
		wy, v = golden(func(y float64) float64 { return k.symbolAt(wx, y) }, wy-half, wy+half)
		if v < best {
			best = v
		}
	}
	return best
}

// symbolDip 是频域扫描出的一个候选低谷。
type symbolDip struct {
	wx, wy, amp float64
}

// insertDip 把候选低谷按振幅升序插入定长列表，只保留最低的
// kernelSymbolSeeds 个。避免对全部 4096 个格点做完整排序。
func insertDip(d []symbolDip, v symbolDip) []symbolDip {
	if len(d) == kernelSymbolSeeds && v.amp >= d[len(d)-1].amp {
		return d
	}
	i := len(d)
	if i > kernelSymbolSeeds-1 {
		i = kernelSymbolSeeds - 1
		d = d[:i]
	}
	d = append(d, symbolDip{})
	for i > 0 && d[i-1].amp > v.amp {
		d[i] = d[i-1]
		i--
	}
	d[i] = v
	return d
}

// apply 用核做二维卷积；adjoint 为真时返回其精确转置。
//
// 正变换取重复边界：越界邻点的权重按 clamp 并入边缘像素，因此映射
// 在边界上是多对一的。此时转置不能简单地把核镜像后同样取 clamp——
// 那只是“相关”，<Kx,y> 与 <x,Kᵀy> 不再相等（非对称核尤为明显）。
// 转置必须反向散射：正变换里 (y,x) 读入 (sy,sx) 的权重加到哪里，
// 转置里就把 y[x] 的贡献加回同一个 (sy,sx)，边界像素因此累计多次。
func (k Kernel) apply(src []float64, w, h int, adjoint bool) []float64 {
	r := k.Size / 2
	out := make([]float64, w*h)
	if adjoint {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := src[y*w+x]
				for ky := 0; ky < k.Size; ky++ {
					sy := clampInt(y+ky-r, 0, h-1)
					row := ky * k.Size
					for kx := 0; kx < k.Size; kx++ {
						sx := clampInt(x+kx-r, 0, w-1)
						out[sy*w+sx] += k.W[row+kx] * v
					}
				}
			}
		}
		return out
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var s float64
			for ky := 0; ky < k.Size; ky++ {
				sy := clampInt(y+ky-r, 0, h-1)
				row := ky * k.Size
				for kx := 0; kx < k.Size; kx++ {
					sx := clampInt(x+kx-r, 0, w-1)
					s += k.W[row+kx] * src[sy*w+sx]
				}
			}
			out[y*w+x] = s
		}
	}
	return out
}

// Blur 用核模糊一张灰度图，四舍五入回 8 位。重复边界。
// 主要供“已知模糊核”的核验构造观测图，也可用于生成对照基线。
func (k Kernel) Blur(g *Gray) (*Gray, error) {
	nk, err := normalizeKernel(k.W, k.Size)
	if err != nil {
		return nil, err
	}
	if err := validateImage(g); err != nil {
		return nil, err
	}
	dst := nk.apply(grayToFloat(g), g.W, g.H, false)
	out, err := New(g.W, g.H)
	if err != nil {
		return nil, err
	}
	for i := range dst {
		out.Pix[i] = clampByte(dst[i])
	}
	return out, nil
}

func validateImage(g *Gray) error {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) < g.W*g.H {
		return &FieldError{Field: "image", Err: ErrEmptyImage}
	}
	return nil
}

func grayToFloat(g *Gray) []float64 {
	out := make([]float64, g.W*g.H)
	for i := range out {
		out[i] = float64(g.Pix[i])
	}
	return out
}
