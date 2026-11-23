package img

import (
	"errors"
	"math"
)

// 放大倍率与输入校验错误。FieldError 的 Error 以字段名开头，便于定位非法参数。
var (
	ErrScale        = errors.New("放大倍率必须为不小于 1 的有限值")
	ErrEmptyImage   = errors.New("图为空")
	ErrGradientZero = errors.New("梯度恒为零，无法解出边缘方向")
)

// FieldError 在错误信息前带上非法字段名。
type FieldError struct {
	Field string
	Err   error
}

func (e *FieldError) Error() string { return e.Field + ": " + e.Err.Error() }
func (e *FieldError) Unwrap() error { return e.Err }

// FieldName 返回非法字段名。
func (e *FieldError) FieldName() string { return e.Field }

// 边缘导向放大的实现常量。取值由 img/upscale_test.go 中五类用例、
// 两档倍率的相似度与过冲核验锁定，改动任一项都应重跑该核验。
const (
	// 沿边缘方向插值取四个采样点（-1.5/-0.5/0.5/1.5 个源像素），
	// 权重为 Keys 三次卷积核（a=-0.5）。
	tangentW0 = -0.0625
	tangentW1 = 0.5625
	tangentW2 = 0.5625
	tangentW3 = -0.0625

	// 沿法向估计局部均值时的采样距离（源图像素）。
	normalOffset = 1.0

	// 结构张量平滑核：半径 2、sigma 1。
	tensorRadius = 2
	tensorSigma  = 1.0

	// 限幅锐化强度。过冲由随后的值域钳制清零，故此处只影响锐度。
	edgeGain = 3.0

	// 相干度低于 1 时向双三次回退的强度：各向同性纹理没有可靠方向，
	// 强行沿切向采样会丢细节。
	edgeBlend = 1.3
)

func validateUpscale(g *Gray, scale float64) error {
	if math.IsNaN(scale) || math.IsInf(scale, 0) || scale < 1 {
		return &FieldError{Field: "scale", Err: ErrScale}
	}
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) < g.W*g.H {
		return &FieldError{Field: "image", Err: ErrEmptyImage}
	}
	return nil
}

func outputSize(n int, scale float64) int {
	m := int(math.Round(float64(n) * scale))
	if m < 1 {
		m = 1
	}
	return m
}

// UpscaleNearest 为最近邻放大基线，接口保持稳定。
// 目标像素取中心映射后的源像素，整数倍率下即块复制。
func UpscaleNearest(g *Gray, scale float64) (*Gray, error) {
	if err := validateUpscale(g, scale); err != nil {
		return nil, err
	}
	ow, oh := outputSize(g.W, scale), outputSize(g.H, scale)
	out, err := New(ow, oh)
	if err != nil {
		return nil, err
	}
	rx := float64(g.W) / float64(ow)
	ry := float64(g.H) / float64(oh)
	for y := 0; y < oh; y++ {
		sy := clampInt(int(math.Floor((float64(y)+0.5)*ry)), 0, g.H-1)
		for x := 0; x < ow; x++ {
			sx := clampInt(int(math.Floor((float64(x)+0.5)*rx)), 0, g.W-1)
			out.Pix[y*ow+x] = g.Pix[sy*g.W+sx]
		}
	}
	return out, nil
}

// cubicWeight 返回 Keys 三次卷积核（a=-0.5）在距离 x 处的值。
func cubicWeight(x float64) float64 {
	x = math.Abs(x)
	const a = -0.5
	if x <= 1 {
		return (a+2)*x*x*x - (a+3)*x*x + 1
	}
	if x < 2 {
		return a*x*x*x - 5*a*x*x + 8*a*x - 4*a
	}
	return 0
}

func sampleBilinear(g *Gray, x, y float64) float64 {
	x = clampFloat(x, 0, float64(g.W-1))
	y = clampFloat(y, 0, float64(g.H-1))
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := clampInt(x0+1, 0, g.W-1)
	y1 := clampInt(y0+1, 0, g.H-1)
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := float64(g.Pix[y0*g.W+x0])
	v10 := float64(g.Pix[y0*g.W+x1])
	v01 := float64(g.Pix[y1*g.W+x0])
	v11 := float64(g.Pix[y1*g.W+x1])
	top := v00*(1-fx) + v10*fx
	bot := v01*(1-fx) + v11*fx
	return top*(1-fy) + bot*fy
}

func sampleBicubic(g *Gray, x, y float64) float64 {
	xi := int(math.Floor(x))
	yi := int(math.Floor(y))
	var sum float64
	for j := -1; j <= 2; j++ {
		wy := cubicWeight(y - float64(yi+j))
		if wy == 0 {
			continue
		}
		sy := clampInt(yi+j, 0, g.H-1)
		for i := -1; i <= 2; i++ {
			wx := cubicWeight(x - float64(xi+i))
			if wx == 0 {
				continue
			}
			sx := clampInt(xi+i, 0, g.W-1)
			sum += wx * wy * float64(g.Pix[sy*g.W+sx])
		}
	}
	return sum
}

// UpscaleBicubic 为双三次放大基线（Keys a=-0.5）。
func UpscaleBicubic(g *Gray, scale float64) (*Gray, error) {
	if err := validateUpscale(g, scale); err != nil {
		return nil, err
	}
	ow, oh := outputSize(g.W, scale), outputSize(g.H, scale)
	out, err := New(ow, oh)
	if err != nil {
		return nil, err
	}
	rx := float64(g.W) / float64(ow)
	ry := float64(g.H) / float64(oh)
	for y := 0; y < oh; y++ {
		v := (float64(y)+0.5)*ry - 0.5
		for x := 0; x < ow; x++ {
			u := (float64(x)+0.5)*rx - 0.5
			out.Pix[y*ow+x] = clampByte(sampleBicubic(g, u, v))
		}
	}
	return out, nil
}

// gradientField 返回源图的 Sobel 梯度平方场，并用高斯核平滑成结构张量。
func gradientField(g *Gray) (jxx, jxy, jyy []float64) {
	w, h := g.W, g.H
	gx := make([]float64, w*h)
	gy := make([]float64, w*h)
	at := func(x, y int) float64 {
		return float64(g.Pix[clampInt(y, 0, h-1)*w+clampInt(x, 0, w-1)])
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gx[y*w+x] = (at(x+1, y-1) + 2*at(x+1, y) + at(x+1, y+1)) -
				(at(x-1, y-1) + 2*at(x-1, y) + at(x-1, y+1))
			gy[y*w+x] = (at(x-1, y+1) + 2*at(x, y+1) + at(x+1, y+1)) -
				(at(x-1, y-1) + 2*at(x, y-1) + at(x+1, y-1))
		}
	}
	jxx = make([]float64, w*h)
	jxy = make([]float64, w*h)
	jyy = make([]float64, w*h)
	for i := range gx {
		jxx[i] = gx[i] * gx[i]
		jxy[i] = gx[i] * gy[i]
		jyy[i] = gy[i] * gy[i]
	}
	k := gaussianKernel()
	smooth := func(src []float64) []float64 {
		tmp := make([]float64, w*h)
		dst := make([]float64, w*h)
		r := len(k) / 2
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
	return smooth(jxx), smooth(jxy), smooth(jyy)
}

func gaussianKernel() []float64 {
	r := tensorRadius
	k := make([]float64, 2*r+1)
	var sum float64
	for i := -r; i <= r; i++ {
		v := math.Exp(-float64(i*i) / (2 * tensorSigma * tensorSigma))
		k[i+r] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// footprintRange 返回中心映射位置 (u,v) 处双线性插值实际触及的源像素值域。
// 权重恰为零的邻点不计入，故整数坐标只取单个像素。
func footprintRange(g *Gray, u, v float64) (float64, float64) {
	u = clampFloat(u, 0, float64(g.W-1))
	v = clampFloat(v, 0, float64(g.H-1))
	x0 := int(math.Floor(u))
	y0 := int(math.Floor(v))
	fx := u - float64(x0)
	fy := v - float64(y0)
	xs := []int{x0}
	ys := []int{y0}
	if fx > 0 && x0+1 < g.W {
		xs = append(xs, x0+1)
	}
	if fy > 0 && y0+1 < g.H {
		ys = append(ys, y0+1)
	}
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, yy := range ys {
		for _, xx := range xs {
			p := float64(g.Pix[yy*g.W+xx])
			if p < lo {
				lo = p
			}
			if p > hi {
				hi = p
			}
		}
	}
	return lo, hi
}

// UpscaleEdgeDirected 为边缘导向放大：
//  1. 由平滑结构张量在目标点解出边缘法向/切向与相干度；
//  2. 沿边缘方向做 Keys 三次插值，避免跨边混合；
//  3. 沿法向做限幅锐化，最后钳到该点双线性的源值域，过冲严格为零。
func UpscaleEdgeDirected(g *Gray, scale float64) (*Gray, error) {
	if err := validateUpscale(g, scale); err != nil {
		return nil, err
	}
	jxx, jxy, jyy := gradientField(g)
	// 梯度恒为零（纯色输入）没有可解的方向，按契约拒绝并报字段名 gradient。
	hasGradient := false
	for i := range jxx {
		if jxx[i] != 0 || jxy[i] != 0 || jyy[i] != 0 {
			hasGradient = true
			break
		}
	}
	if !hasGradient {
		return nil, &FieldError{Field: "gradient", Err: ErrGradientZero}
	}

	ow, oh := outputSize(g.W, scale), outputSize(g.H, scale)
	out, err := New(ow, oh)
	if err != nil {
		return nil, err
	}
	rx := float64(g.W) / float64(ow)
	ry := float64(g.H) / float64(oh)
	for y := 0; y < oh; y++ {
		v := (float64(y)+0.5)*ry - 0.5
		for x := 0; x < ow; x++ {
			u := (float64(x)+0.5)*rx - 0.5
			out.Pix[y*ow+x] = clampByte(edgeDirectedValue(g, jxx, jxy, jyy, u, v))
		}
	}
	return out, nil
}

// gradientAt 在源坐标 (u,v) 处双线性采样结构张量。
func gradientAt(g *Gray, jxx, jxy, jyy []float64, u, v float64) (a, b, c float64) {
	u = clampFloat(u, 0, float64(g.W-1))
	v = clampFloat(v, 0, float64(g.H-1))
	x0 := int(math.Floor(u))
	y0 := int(math.Floor(v))
	x1 := clampInt(x0+1, 0, g.W-1)
	y1 := clampInt(y0+1, 0, g.H-1)
	fx := u - float64(x0)
	fy := v - float64(y0)
	lerp := func(s []float64) float64 {
		v00 := s[y0*g.W+x0]
		v10 := s[y0*g.W+x1]
		v01 := s[y1*g.W+x0]
		v11 := s[y1*g.W+x1]
		return (v00*(1-fx)+v10*fx)*(1-fy) + (v01*(1-fx)+v11*fx)*fy
	}
	return lerp(jxx), lerp(jxy), lerp(jyy)
}

func edgeDirectedValue(g *Gray, jxx, jxy, jyy []float64, u, v float64) float64 {
	lo, hi := footprintRange(g, u, v)
	a, b, c := gradientAt(g, jxx, jxy, jyy, u, v)
	tr := a + c
	disc := math.Sqrt(math.Max(0, (a-c)*(a-c)+4*b*b))
	l1 := (tr + disc) / 2
	l2 := (tr - disc) / 2
	coherence := 0.0
	if l1+l2 > 1e-12 {
		coherence = (l1 - l2) / (l1 + l2)
	}

	var nx, ny float64
	if math.Abs(b) < 1e-12 {
		if a >= c {
			nx = 1
		} else {
			ny = 1
		}
	} else {
		nx, ny = b, l1-a
		nrm := math.Hypot(nx, ny)
		if nrm < 1e-12 {
			nx, ny = 1, 0
		} else {
			nx, ny = nx/nrm, ny/nrm
		}
	}
	tx, ty := -ny, nx

	// 沿边缘方向三次插值：四个采样点位于目标点两侧的切线上。
	along := tangentW0*sampleBilinear(g, u-1.5*tx, v-1.5*ty) +
		tangentW1*sampleBilinear(g, u-0.5*tx, v-0.5*ty) +
		tangentW2*sampleBilinear(g, u+0.5*tx, v+0.5*ty) +
		tangentW3*sampleBilinear(g, u+1.5*tx, v+1.5*ty)

	// 基底插值：沿边缘方向采用三次核，相干度低处混合双三次，
	// 各向同性纹理没有可靠方向，强行沿切向采样只会丢细节。
	base := along
	if edgeBlend > 0 {
		iso := sampleBicubic(g, u, v)
		w := edgeBlend * (1 - coherence)
		if w > 1 {
			w = 1
		}
		base = (1-w)*along + w*iso
	}

	// 沿法向估计局部均值，对基底做限幅锐化；相干度低处刃度自动减弱。
	fp := sampleBilinear(g, u+normalOffset*nx, v+normalOffset*ny)
	fm := sampleBilinear(g, u-normalOffset*nx, v-normalOffset*ny)
	val := base + edgeGain*coherence*(base-0.5*(fp+fm))

	// 限幅：结果不得越出该点双线性插值实际触及的源像素值域。
	if val < lo {
		val = lo
	}
	if val > hi {
		val = hi
	}
	return val
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampByte(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
