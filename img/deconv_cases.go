package img

import "math"

// 五幅合成用例：阶跃、细线、棋盘、噪点、纯色。
// 每幅先由 clean 构造，再用已知核模糊得到观测图，作为反卷积输入。
// 固定尺寸与随机种子，保证核验可复现。

// DeconvCase 描述一幅核验用例。
type DeconvCase struct {
	Name string
	// Clean 是未模糊的源图。
	Clean *Gray
	// Observed 是用已知核模糊后的观测图，即反卷积的输入。
	Observed *Gray
}

// DeconvCaseNames 是五类用例的固定顺序。
var DeconvCaseNames = []string{"阶跃", "细线", "棋盘", "噪点", "纯色"}

const deconvCaseSize = 128

// DeconvCases 用给定核生成五幅用例。核须通过 NewKernel/GaussianKernel 的校验。
func DeconvCases(k Kernel) ([]DeconvCase, error) {
	nk, err := checkKernel(k)
	if err != nil {
		return nil, err
	}
	clean := deconvCleanImages()
	out := make([]DeconvCase, 0, len(clean))
	for _, c := range clean {
		obs, err := nk.Blur(c.Clean)
		if err != nil {
			return nil, err
		}
		out = append(out, DeconvCase{Name: c.Name, Clean: c.Clean, Observed: obs})
	}
	return out, nil
}

func deconvCleanImages() []DeconvCase {
	const n = deconvCaseSize
	build := func(name string, f func(x, y int) float64) DeconvCase {
		g, _ := New(n, n)
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				g.Pix[y*n+x] = clampByte(f(x, y))
			}
		}
		return DeconvCase{Name: name, Clean: g}
	}
	// 噪点用固定种子的线性同余，避免依赖 math/rand 的版本行为。
	seed := uint32(20261124)
	next := func() float64 {
		seed = seed*1664525 + 1013904223
		return float64(seed >> 24) // 高 8 位，0..255
	}
	noise := make([]uint8, n*n)
	for i := range noise {
		noise[i] = uint8(next())
	}
	return []DeconvCase{
		build("阶跃", func(x, _ int) float64 {
			if x < n/2 {
				return 30
			}
			return 220
		}),
		build("细线", func(_, y int) float64 {
			if y >= n/2-1 && y <= n/2+1 {
				return 230
			}
			return 35
		}),
		build("棋盘", func(x, y int) float64 {
			if ((x/8)+(y/8))%2 == 0 {
				return 40
			}
			return 210
		}),
		DeconvCase{Name: "噪点", Clean: &Gray{W: n, H: n, Pix: noise}},
		build("纯色", func(_, _ int) float64 { return 128 }),
	}
}

// 后置校验辅助：供测试与报表共用，避免两处判据漂移。

// RingStats 返回图 x 相对参考图同位置 3×3 局部值域的振铃幅度与振铃像素数。
// 参考图用观测图 g 而非未模糊源图：核验只用反卷积实际拿到的信息。
func RingStats(x, ref *Gray) (float64, int) {
	if x == nil || ref == nil || x.W != ref.W || x.H != ref.H || x.W <= 0 {
		return math.Inf(1), -1
	}
	lo, hi := localRange3x3(grayToFloat(ref), ref.W, ref.H)
	return ringStats(grayToFloat(x), lo, hi)
}

// Sum 返回整图灰度总和。
func Sum(g *Gray) float64 {
	if g == nil || g.W <= 0 || g.H <= 0 {
		return math.NaN()
	}
	var s float64
	for _, v := range g.Pix {
		s += float64(v)
	}
	return s
}

// MinPix 返回整图最小灰度。
func MinPix(g *Gray) int {
	if g == nil || len(g.Pix) == 0 {
		return -1
	}
	m := 255
	for _, v := range g.Pix {
		if int(v) < m {
			m = int(v)
		}
	}
	return m
}

// RelResidual 返回 ||K·x − g||₂ / ||g||₂，即相对 L2 残差。
func RelResidual(x, g *Gray, k Kernel) (float64, error) {
	if err := validateImage(x); err != nil {
		return 0, err
	}
	if err := validateImage(g); err != nil {
		return 0, err
	}
	if x.W != g.W || x.H != g.H {
		return 0, &FieldError{Field: "image", Err: ErrEmptyImage}
	}
	nk, err := checkKernel(k)
	if err != nil {
		return 0, err
	}
	xf := grayToFloat(x)
	gf := grayToFloat(g)
	norm := math.Sqrt(dot(gf, gf))
	if norm == 0 {
		return 0, nil
	}
	d := sub(nk.apply(xf, x.W, x.H, false), gf)
	return math.Sqrt(dot(d, d)) / norm, nil
}
