package img

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// 误差扩散抖动相关的具名错误。调用方可以用 errors.Is 判定。
var (
	// ErrKernel 表示扩散核参数非法（空核、权重表不完整、权重非正、
	// 权重和越界、偏移不前进或重复等）。
	ErrKernel = errors.New("扩散核非法")
	// ErrStrength 表示扩散强度非法（负数、NaN 或无穷）。
	ErrStrength = errors.New("扩散强度非法")
	// ErrBits 表示位深越界（只接受 1 到 8）。
	ErrBits = errors.New("位深越界")
)

// Kernel 是一张误差扩散权重表。DX/DY 是以当前像素为原点的整数偏移，
// W 是与偏移一一对应的权重。扫描方向为正时按 (DX,DY) 扩散；
// 蛇形反向行会把 DX 镜像为 -DX。
type Kernel struct {
	Name string
	DX   []int
	DY   []int
	W    []float64
}

// Validate 检查扩散核是否完整、合法。
func (k *Kernel) Validate() error {
	if k == nil {
		return fmt.Errorf("%w: 核为空", ErrKernel)
	}
	if strings.TrimSpace(k.Name) == "" {
		return fmt.Errorf("%w: 核名为空", ErrKernel)
	}
	n := len(k.W)
	if n == 0 {
		return fmt.Errorf("%w: %q 权重表为空", ErrKernel, k.Name)
	}
	if len(k.DX) != n || len(k.DY) != n {
		return fmt.Errorf("%w: %q 偏移与权重项数不一致（%d/%d/%d）",
			ErrKernel, k.Name, len(k.DX), len(k.DY), n)
	}
	seen := make(map[[2]int]bool, n)
	sum := 0.0
	for i := 0; i < n; i++ {
		dx, dy, w := k.DX[i], k.DY[i], k.W[i]
		if math.IsNaN(w) || math.IsInf(w, 0) || w <= 0 {
			return fmt.Errorf("%w: %q 第 %d 项权重必须为正有限数，实际 %v",
				ErrKernel, k.Name, i, w)
		}
		if dx == 0 && dy == 0 {
			return fmt.Errorf("%w: %q 第 %d 项偏移为原点", ErrKernel, k.Name, i)
		}
		if dy < 0 || (dy == 0 && dx <= 0) {
			return fmt.Errorf("%w: %q 第 %d 项偏移 (%d,%d) 不在扫描前进方向",
				ErrKernel, k.Name, i, dx, dy)
		}
		key := [2]int{dx, dy}
		if seen[key] {
			return fmt.Errorf("%w: %q 偏移 (%d,%d) 重复", ErrKernel, k.Name, dx, dy)
		}
		seen[key] = true
		sum += w
	}
	if sum <= 0 || sum > 1+1e-9 {
		return fmt.Errorf("%w: %q 权重和为 %v，必须在 (0,1] 内", ErrKernel, k.Name, sum)
	}
	return nil
}

type kernelTap struct {
	dx, dy int
	num    int
	den    int
}

func newKernel(name string, taps ...kernelTap) *Kernel {
	k := &Kernel{Name: name}
	for _, t := range taps {
		k.DX = append(k.DX, t.dx)
		k.DY = append(k.DY, t.dy)
		k.W = append(k.W, float64(t.num)/float64(t.den))
	}
	return k
}

// 内置核的权重表按“同一项偏移 + 权重”逐项定义，测试会逐项核对。
var builtinKernels = []*Kernel{
	newKernel("floyd-steinberg",
		kernelTap{1, 0, 7, 16},
		kernelTap{-1, 1, 3, 16},
		kernelTap{0, 1, 5, 16},
		kernelTap{1, 1, 1, 16},
	),
	newKernel("jarvis-judice-ninke",
		kernelTap{1, 0, 7, 48},
		kernelTap{2, 0, 5, 48},
		kernelTap{-2, 1, 3, 48},
		kernelTap{-1, 1, 5, 48},
		kernelTap{0, 1, 7, 48},
		kernelTap{1, 1, 5, 48},
		kernelTap{2, 1, 3, 48},
		kernelTap{-2, 2, 1, 48},
		kernelTap{-1, 2, 3, 48},
		kernelTap{0, 2, 5, 48},
		kernelTap{1, 2, 3, 48},
		kernelTap{2, 2, 1, 48},
	),
	newKernel("stucki",
		kernelTap{1, 0, 8, 42},
		kernelTap{2, 0, 4, 42},
		kernelTap{-2, 1, 2, 42},
		kernelTap{-1, 1, 4, 42},
		kernelTap{0, 1, 8, 42},
		kernelTap{1, 1, 4, 42},
		kernelTap{2, 1, 2, 42},
		kernelTap{-2, 2, 1, 42},
		kernelTap{-1, 2, 2, 42},
		kernelTap{0, 2, 4, 42},
		kernelTap{1, 2, 2, 42},
		kernelTap{2, 2, 1, 42},
	),
	newKernel("atkinson",
		kernelTap{1, 0, 1, 8},
		kernelTap{2, 0, 1, 8},
		kernelTap{-1, 1, 1, 8},
		kernelTap{0, 1, 1, 8},
		kernelTap{1, 1, 1, 8},
		kernelTap{0, 2, 1, 8},
	),
	newKernel("burkes",
		kernelTap{1, 0, 8, 32},
		kernelTap{2, 0, 4, 32},
		kernelTap{-2, 1, 2, 32},
		kernelTap{-1, 1, 4, 32},
		kernelTap{0, 1, 8, 32},
		kernelTap{1, 1, 4, 32},
		kernelTap{2, 1, 2, 32},
	),
	newKernel("sierra",
		kernelTap{1, 0, 5, 32},
		kernelTap{2, 0, 3, 32},
		kernelTap{-2, 1, 2, 32},
		kernelTap{-1, 1, 4, 32},
		kernelTap{0, 1, 5, 32},
		kernelTap{1, 1, 4, 32},
		kernelTap{2, 1, 2, 32},
		kernelTap{-1, 2, 2, 32},
		kernelTap{0, 2, 3, 32},
		kernelTap{1, 2, 2, 32},
	),
	newKernel("sierra-lite",
		kernelTap{1, 0, 2, 4},
		kernelTap{-1, 1, 1, 4},
		kernelTap{0, 1, 1, 4},
	),
}

// BuiltinKernels 返回内置扩散核的副本，顺序固定。
func BuiltinKernels() []*Kernel {
	out := make([]*Kernel, len(builtinKernels))
	for i, k := range builtinKernels {
		out[i] = k.clone()
	}
	return out
}

// LookupKernel 按名字取内置核，名字不区分大小写。
func LookupKernel(name string) (*Kernel, error) {
	n := strings.ToLower(strings.TrimSpace(name))
	n = strings.ReplaceAll(n, "_", "-")
	for _, k := range builtinKernels {
		if k.Name == n {
			return k.clone(), nil
		}
	}
	return nil, fmt.Errorf("%w: 未知核 %q", ErrKernel, name)
}

func (k *Kernel) clone() *Kernel {
	if k == nil {
		return nil
	}
	return &Kernel{
		Name: k.Name,
		DX:   append([]int(nil), k.DX...),
		DY:   append([]int(nil), k.DY...),
		W:    append([]float64(nil), k.W...),
	}
}

// DitherOptions 控制误差扩散抖动。
type DitherOptions struct {
	// Bits 是目标位深，1..8。8 表示直通，输出与输入逐字节相同。
	Bits int
	// Strength 是扩散强度，必须非负。0 表示不扩散误差，只做最近色阶量化。
	Strength float64
	// Kernel 是扩散核。Bits 小于 8 且 Strength 大于 0 时必须提供。
	Kernel *Kernel
}

// Dither 用蛇形推进的误差扩散把 Gray 量化到 1<<Bits 个色阶。
//
// 扫描顺序逐行蛇形：偶数行从左到右，奇数行从右到左，反向行把核的 DX
// 镜像。Bits=8 是直通路径，输出与输入逐字节一致；这时不要求扩散核。
// Bits<8 且 Strength=0 时只做最近色阶量化、不扩散误差。
func Dither(g *Gray, opt DitherOptions) (*Gray, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) != g.W*g.H {
		return nil, ErrSize
	}
	if opt.Bits < 1 || opt.Bits > 8 {
		return nil, fmt.Errorf("%w: %d（允许 1 到 8）", ErrBits, opt.Bits)
	}
	if math.IsNaN(opt.Strength) || math.IsInf(opt.Strength, 0) || opt.Strength < 0 {
		return nil, fmt.Errorf("%w: %v（必须为非负有限数）", ErrStrength, opt.Strength)
	}
	if opt.Kernel != nil {
		if err := opt.Kernel.Validate(); err != nil {
			return nil, err
		}
	}

	out, err := New(g.W, g.H)
	if err != nil {
		return nil, err
	}
	// 直通：保持原样，一个字节都不动。
	if opt.Bits == 8 {
		copy(out.Pix, g.Pix)
		return out, nil
	}
	if opt.Strength > 0 && opt.Kernel == nil {
		return nil, fmt.Errorf("%w: 强度为正但未提供扩散核", ErrKernel)
	}

	levels := 1 << opt.Bits
	pal := paletteForLevels(levels)
	step := paletteStep(levels)
	w, h := g.W, g.H
	errBuf := make([]float64, w*h)

	for y := 0; y < h; y++ {
		for stepX := 0; stepX < w; stepX++ {
			dir := 1
			x := stepX
			if y%2 == 1 {
				dir = -1
				x = w - 1 - stepX
			}
			i := y*w + x
			cur := float64(g.Pix[i]) + errBuf[i]
			q := pal[nearestPaletteIndex(pal, cur, step)]
			out.Pix[i] = q

			if opt.Strength == 0 || opt.Kernel == nil {
				continue
			}
			e := (cur - float64(q)) * opt.Strength
			if e == 0 {
				continue
			}
			for t := range opt.Kernel.W {
				nx := x + dir*opt.Kernel.DX[t]
				ny := y + opt.Kernel.DY[t]
				if nx < 0 || nx >= w || ny < 0 || ny >= h {
					continue
				}
				errBuf[ny*w+nx] += e * opt.Kernel.W[t]
			}
		}
	}
	return out, nil
}

// paletteStep 返回相邻色阶的步长。1 色阶时整幅图归零，步长无意义，记 255。
func paletteStep(levels int) float64 {
	if levels <= 1 {
		return 255
	}
	return float64(255 / (levels - 1))
}

// paletteForLevels 与 Quantize 使用同一套色阶：k*step。
func paletteForLevels(levels int) []uint8 {
	pal := make([]uint8, levels)
	step := paletteStep(levels)
	for k := 0; k < levels; k++ {
		v := float64(k) * step
		if v > 255 {
			v = 255
		}
		pal[k] = uint8(v)
	}
	return pal
}

func nearestPaletteIndex(pal []uint8, v, step float64) int {
	last := len(pal) - 1
	if v <= float64(pal[0]) {
		return 0
	}
	if v >= float64(pal[last]) {
		return last
	}
	k := int(math.Round(v / step))
	if k < 0 {
		k = 0
	}
	if k > last {
		k = last
	}
	best := k
	bestDist := math.Abs(v - float64(pal[k]))
	if k > 0 {
		if d := math.Abs(v - float64(pal[k-1])); d < bestDist {
			best, bestDist = k-1, d
		}
	}
	if k < last {
		if d := math.Abs(v - float64(pal[k+1])); d < bestDist {
			best = k + 1
		}
	}
	return best
}

// MeanAbsError 返回两幅同尺寸图的平均绝对误差。
func MeanAbsError(a, b *Gray) (float64, error) {
	if a == nil || b == nil || a.W != b.W || a.H != b.H || len(a.Pix) != len(b.Pix) {
		return 0, fmt.Errorf("图像尺寸不一致")
	}
	if len(a.Pix) == 0 {
		return 0, ErrEmpty
	}
	sum := 0.0
	for i := range a.Pix {
		d := float64(a.Pix[i]) - float64(b.Pix[i])
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return sum / float64(len(a.Pix)), nil
}
