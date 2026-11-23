package img

import (
	"errors"
	"math"
)

// Filter 选择重采样核。
type Filter int

const (
	FilterNearest  Filter = iota // 最近邻：盒核
	FilterBilinear               // 双线性：三角核
	FilterBicubic                // 双三次：Keys 三次卷积核（A=-0.5）
)

// DefaultAliasBound 是目标频谱允许的混叠幅度上界 ε 的默认值。
const DefaultAliasBound = 1e-2

var (
	ErrParamAngle  = errors.New("参数 angle 非法：角度必须为有限值，支持任意有限角度旋转")
	ErrParamScale  = errors.New("参数 scale 非法：缩放必须为正数，不能为零或负")
	ErrParamImage  = errors.New("参数 image 非法：图为空")
	ErrParamFilter = errors.New("参数 filter 非法：未知的重采样方式")
)

// KernelWidth 由目标频谱的混叠上界 aliasBound 反解重采样核相对标称核的
// 拉伸倍数 w（w>=1）。
//
// 推导：源图频谱以 2π 为周期延拓，压缩比 m=min(1,scale) 时预过滤截止频率
// 为 ωc=πm；采样后相邻镜像频谱从 ωc 处即进入通带，混叠带边沿 ωa=ωc。
// 要求拉伸后的核频谱在整条混叠带 |ω|>=ωa 内不超过 aliasBound，据此反解 w。
// 放大（scale>=1）时无混叠，w=1，使用标称核即可。
func KernelWidth(f Filter, scale, aliasBound float64) float64 {
	if !(aliasBound > 0) {
		aliasBound = DefaultAliasBound
	}
	m := scale
	if m > 1 {
		m = 1
	}
	if m >= 1 {
		return 1
	}
	// 源频谱的镜像中心在 ±2π，预过滤截止频率 ωc=πm；采样后镜像
	// 频谱从 ωc 处即进入通带，故混叠带边沿取 ωa=ωc=πm。
	// 要求拉伸后的核频谱在 |ω|>=ωa 整条混叠带内不超过 aliasBound。
	wa := math.Pi * m
	switch f {
	case FilterNearest:
		// 标称盒核半宽 0.5，拉伸 w 后 H(ω)=sinc(0.5·w·ω)。
		// 利用 |sinc(z)|<=1/|z|，要求 2/(w·ωa)<=ε，闭式反解。
		w := 2 / (aliasBound * wa)
		return math.Max(1, w)
	case FilterBilinear:
		// 标称三角核半宽 1，拉伸 w 后 H(ω)=sinc²(w·ω)。
		// 包络 1/(w·ωa)²<=ε，闭式反解。
		w := 1 / (wa * math.Sqrt(aliasBound))
		return math.Max(1, w)
	case FilterBicubic:
		return cubicKernelWidth(wa, aliasBound)
	default:
		return 1
	}
}

// radiusFromW 返回给定核拉伸倍数下、源图坐标中的支撑半径。
func radiusFromW(f Filter, w float64) float64 {
	switch f {
	case FilterNearest:
		return 0.5 * w
	case FilterBilinear:
		return w
	default:
		return 2 * w
	}
}

// KernelRadius 返回解出的核在源图像素坐标系下的支撑半径（半宽）。
func KernelRadius(f Filter, scale, aliasBound float64) float64 {
	return radiusFromW(f, KernelWidth(f, scale, aliasBound))
}

// cubicBase 是 Keys 三次卷积核（A=-0.5），支撑 |x|<=2，整数采样处构成单位分解。
func cubicBase(x float64) float64 {
	if x < 0 {
		x = -x
	}
	const A = -0.5
	switch {
	case x <= 1:
		return (A+2)*x*x*x - (A+3)*x*x + 1
	case x < 2:
		return A*x*x*x - 5*A*x*x + 8*A*x - 4*A
	default:
		return 0
	}
}

// polyCosInt 求多项式 c0+c1·x+c2·x²+c3·x³ 与 cos(ωx) 之积在 [a,b] 上的定积分。
func polyCosInt(c0, c1, c2, c3, a, b, omega float64) float64 {
	anti := func(x float64) float64 {
		sn, cs := math.Sincos(omega * x)
		w, w2, w3, w4 := omega, omega*omega, omega*omega*omega, omega*omega*omega*omega
		i0 := sn / w
		i1 := x*sn/w + cs/w2
		i2 := x*x*sn/w + 2*x*cs/w2 - 2*sn/w3
		i3 := x*x*x*sn/w + 3*x*x*cs/w2 - 6*x*sn/w3 - 6*cs/w4
		return c0*i0 + c1*i1 + c2*i2 + c3*i3
	}
	return anti(b) - anti(a)
}

// cubicFT 是三次卷积核的傅里叶变换（实偶函数，仅余弦分量）。
func cubicFT(omega float64) float64 {
	if omega == 0 {
		return 1 // 核积分为 1
	}
	const A = -0.5
	v := polyCosInt(1, 0, -(A + 3), A+2, 0, 1, omega)
	v += polyCosInt(-4*A, 8*A, -5*A, A, 1, 2, omega)
	return 2 * v
}

// cubicKernelWidth 数值反解三次核拉伸倍数：在混叠带网格上取频率响应最大模，
// 二分求满足最大模 <= ε 的最小 w。
func cubicKernelWidth(wa, eps float64) float64 {
	const (
		omegaMax = 6 * math.Pi // 更高旁瓣衰减到可忽略，截断于此
		grid     = 512
	)
	maxResp := func(w float64) float64 {
		m := 0.0
		for i := 0; i <= grid; i++ {
			om := wa + (omegaMax-wa)*float64(i)/grid
			if v := math.Abs(cubicFT(w * om)); v > m {
				m = v
			}
		}
		return m
	}
	if maxResp(1) <= eps {
		return 1
	}
	lo, hi := 1.0, 2.0
	for maxResp(hi) > eps {
		hi *= 2
		if hi > 1e6 {
			break
		}
	}
	for i := 0; i < 80; i++ {
		mid := (lo + hi) / 2
		if maxResp(mid) <= eps {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

// kernelEval 按拉伸倍数 w 评估一维核。
func kernelEval(f Filter, w, x float64) float64 {
	ax := math.Abs(x)
	switch f {
	case FilterNearest:
		r := 0.5 * w
		if ax < r {
			return 1 / (2 * r)
		}
		// 标称盒核（w=1）的整数对齐点在 resampleFloat 的 nearestPixel
		// 快速路径处理；宽盒核（w>1）不会在整数距离处取到零权重。
		return 0
	case FilterBilinear:
		r := w
		if ax <= r {
			return (1 / r) * (1 - ax/r)
		}
		return 0
	default:
		r := 2 * w
		if ax <= r {
			return cubicBase(x/w) / w
		}
		return 0
	}
}

type tap struct {
	idx int
	wt  float64
}

// buildTaps 收集中心 center 周围核支撑内的整数采样点。
//
// 边界处理按场景分流：
//   - clamp=true（纯缩放，无背景概念）：图外核权重钳制到最近边缘像素，
//     分母只累加图内（含钳制）权重——replicate padding 加自归一。
//     单位分解核在该模式下离散总质量守恒；宽预过滤核的图外权重并入边缘
//     像素而非稀释平均值，小图强压缩也不会丢质量。
//   - clamp=false（旋转，存在真背景）：图外权重不参与累加但计入分母
//     （零填充归一），边缘按覆盖率自然衰减为背景零。
//
// 完全无图内抽头时返回空列表，由调用方置背景零。
func buildTaps(f Filter, w, center float64, size int, clamp bool) ([]tap, float64) {
	r := radiusFromW(f, w)
	lo := int(math.Floor(center - r))
	hi := int(math.Ceil(center + r))
	taps := make([]tap, 0, hi-lo+1)
	denom := 0.0
	for i := lo; i <= hi; i++ {
		wt := kernelEval(f, w, float64(i)-center)
		if wt == 0 {
			continue
		}
		inBounds := i >= 0 && i < size
		if !clamp {
			// 零填充：分母计全部权重，抽头只留图内。
			denom += wt
			if inBounds {
				taps = append(taps, tap{i, wt})
			}
			continue
		}
		// 钳制自归一：图外索引折到边缘，分母只计这些（图内+钳制）权重。
		ci := i
		if ci < 0 {
			ci = 0
		} else if ci >= size {
			ci = size - 1
		}
		denom += wt
		if pos, ok := tapIndex(taps, ci); ok {
			taps[pos].wt += wt
		} else {
			taps = append(taps, tap{ci, wt})
		}
	}
	// 标称盒核（w=1）理论上在整数对齐位置核值为零，但该情形由
	// resampleFloat 的 nearestPixel 快速路径接管，不会走到这里；
	// 此回退仅作防御性兜底。
	if f == FilterNearest && w == 1 && denom == 0 {
		j := int(math.Floor(center + 0.5))
		if j >= 0 && j < size {
			wt := kernelEval(f, w, 0)
			taps = append(taps, tap{j, wt})
			denom = wt
		}
	}
	return taps, denom
}

// tapIndex 在已收集抽头中查找像素索引 ci 的位置。
func tapIndex(taps []tap, ci int) (int, bool) {
	for k := range taps {
		if taps[k].idx == ci {
			return k, true
		}
	}
	return 0, false
}

// nearestPixel 返回连续坐标 (sx,sy) 的最近整数像素索引。
// 像素 i 覆盖连续坐标 [i, i+1)、中心在 i+0.5，故逆映射坐标 c 的最近
// 索引为 Floor(c)；索引越界表示该点落在源图四边形之外（旋转真背景）。
// 该规则使整数倍 90° 旋转时源/目标像素格一一对应、无重复无遗漏。
func nearestPixel(sx, sy float64, w, h int) (int, int, bool) {
	axis := func(c float64, size int) (int, bool) {
		j := int(math.Floor(c))
		if j < 0 || j >= size {
			return 0, false
		}
		return j, true
	}
	ix, okx := axis(sx, w)
	iy, oky := axis(sy, h)
	return ix, iy, okx && oky
}

func validate(g *Gray, angleDeg, scale float64, f Filter) (float64, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return 0, ErrParamImage
	}
	if math.IsNaN(angleDeg) || math.IsInf(angleDeg, 0) {
		return 0, ErrParamAngle
	}
	if math.IsNaN(scale) || math.IsInf(scale, 0) || scale <= 0 {
		return 0, ErrParamScale
	}
	switch f {
	case FilterNearest, FilterBilinear, FilterBicubic:
	default:
		return 0, ErrParamFilter
	}
	a := math.Mod(angleDeg, 360)
	if a < 0 {
		a += 360
	}
	return a, nil
}

// resampleFloat 是核心实现，返回浮点结果与画幅，量化前用于亮度守恒核算。
func resampleFloat(g *Gray, angleDeg, scale float64, f Filter, aliasBound float64) ([]float64, int, int, error) {
	a, err := validate(g, angleDeg, scale, f)
	if err != nil {
		return nil, 0, 0, err
	}

	// 原尺寸直通：零角度、单位缩放时输出与输入逐像素一致。
	if a == 0 && scale == 1 {
		out := make([]float64, len(g.Pix))
		for i, v := range g.Pix {
			out[i] = float64(v)
		}
		return out, g.W, g.H, nil
	}

	rad := a * math.Pi / 180
	cosA, sinA := math.Cos(rad), math.Sin(rad)
	fw, fh := float64(g.W), float64(g.H)

	// 减容差后再 ceil：吸收 sin(π) 等 1e-16 量级的浮点误差，
	// 保证整倍数 90° 时画幅恰为旋转后的整数尺寸。
	outW := int(math.Ceil(scale*(fw*math.Abs(cosA)+fh*math.Abs(sinA)) - 1e-9))
	outH := int(math.Ceil(scale*(fw*math.Abs(sinA)+fh*math.Abs(cosA)) - 1e-9))
	if outW < 1 {
		outW = 1
	}
	if outH < 1 {
		outH = 1
	}

	w := KernelWidth(f, scale, aliasBound)
	dst := make([]float64, outW*outH)
	// 统一角点坐标框架：像素格为 [x,x+1]×[y,y+1]，两画幅中心均为 (W/2,H/2)，
	// 这样整数倍 90° 旋转时像素格严格对齐，不引入半像素错位。
	ocx, ocy := float64(outW)/2, float64(outH)/2

	// 宽盒核（最近邻强压缩）用积分图做 O(1) 窗口均值，避免逐抽头扫描。
	var rowSum [][]float64
	if f == FilterNearest && w > 1 {
		rowSum = boxIntegralImages(g)
	}

	for oy := 0; oy < outH; oy++ {
		vy := float64(oy) + 0.5 - ocy
		for ox := 0; ox < outW; ox++ {
			vx := float64(ox) + 0.5 - ocx
			// 逆映射：先除以缩放，再反向旋转回源图坐标。
			ux := (cosA*vx + sinA*vy) / scale
			uy := (-sinA*vx + cosA*vy) / scale
			sx, sy := fw/2+ux, fh/2+uy

			// 标称最近邻（w=1，即放大/直通/旋转）：直接取最近像素，
			// 独立于核权重路径以避开盒核边界的零权重歧义。
			if f == FilterNearest && w == 1 {
				if ix, iy, ok := nearestPixel(sx, sy, g.W, g.H); ok {
					dst[oy*outW+ox] = float64(g.Pix[iy*g.W+ix])
				}
				continue
			}

			// 宽盒核：积分图窗口均值。旋转时窗口可能越界，越界部分按
			// 零填充（与 buildTaps 零填充路径一致）。
			if f == FilterNearest && w > 1 {
				dst[oy*outW+ox] = boxWindowMean(rowSum, g, sx, sy, w, a != 0)
				continue
			}

			// 纯缩放无背景，钳制自归一；旋转有真背景，零填充。
			clamp := a == 0
			xt, xden := buildTaps(f, w, sx, g.W, clamp)
			yt, yden := buildTaps(f, w, sy, g.H, clamp)
			denom := xden * yden
			// 任一轴无图内抽头：逆映射点落在源图四边形之外，置背景零。
			if denom == 0 || len(xt) == 0 || len(yt) == 0 {
				continue
			}
			num := 0.0
			for _, t := range yt {
				row := t.idx * g.W
				for _, u := range xt {
					num += t.wt * u.wt * float64(g.Pix[row+u.idx])
				}
			}
			dst[oy*outW+ox] = num / denom
		}
	}
	return dst, outW, outH, nil
}

// boxIntegralImages 构建行方向前缀和，供宽盒核窗口均值 O(1) 求值。
// rowSum[y][x] 是第 y 行前 x 个像素之和。
func boxIntegralImages(g *Gray) [][]float64 {
	rowSum := make([][]float64, g.H)
	for y := 0; y < g.H; y++ {
		rowSum[y] = make([]float64, g.W+1)
		for x := 0; x < g.W; x++ {
			rowSum[y][x+1] = rowSum[y][x] + float64(g.Pix[y*g.W+x])
		}
	}
	return rowSum
}

// boxWindowMean 用积分图求以 (cx,cy) 为中心、半宽 0.5·w 的盒核窗口均值。
// zeroFill=true（旋转）时越界像素按零计、按全窗口面积归一；
// 否则（纯缩放）图外部分钳制到边缘像素，按图内窗口面积自归一。
func boxWindowMean(rowSum [][]float64, g *Gray, cx, cy, w float64, zeroFill bool) float64 {
	r := 0.5 * w
	x0 := cx - r
	x1 := cx + r
	y0 := cy - r
	y1 := cy + r

	// 落入图内的整数像素索引区间（核在这些点上采样）。
	ix0 := int(math.Ceil(x0))
	ix1 := int(math.Floor(x1 - 1e-12))
	iy0 := int(math.Ceil(y0))
	iy1 := int(math.Floor(y1 - 1e-12))

	if zeroFill {
		// 与图无交集：真背景。
		if ix1 < 0 || ix0 >= g.W || iy1 < 0 || iy0 >= g.H {
			return 0
		}
		cx0, cx1 := maxInt(ix0, 0), minInt(ix1, g.W-1)
		cy0, cy1 := maxInt(iy0, 0), minInt(iy1, g.H-1)
		sum := 0.0
		for y := cy0; y <= cy1; y++ {
			sum += rowSum[y][cx1+1] - rowSum[y][cx0]
		}
		// 全窗口面积（含图外零），与零填充分母一致。
		full := float64(ix1-ix0+1) * float64(iy1-iy0+1)
		if full <= 0 {
			return 0
		}
		return sum / full
	}

	// 钳制自归一：索引折到边缘，按图内窗口面积归一。
	cx0, cx1 := clampRange(ix0, ix1, g.W)
	cy0, cy1 := clampRange(iy0, iy1, g.H)
	sum := 0.0
	for y := cy0; y <= cy1; y++ {
		sum += rowSum[y][cx1+1] - rowSum[y][cx0]
	}
	area := float64(cx1-cx0+1) * float64(cy1-cy0+1)
	if area <= 0 {
		return 0
	}
	return sum / area
}

func clampRange(lo, hi, size int) (int, int) {
	if lo < 0 {
		lo = 0
	}
	if hi >= size {
		hi = size - 1
	}
	return lo, hi
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// RotateScale 围绕图像中心做任意角度旋转（角度，度）与等比缩放，
// 重采样核由 f 指定。缩放（scale<1）时按混叠上界反解的核宽做预过滤。
func RotateScale(g *Gray, angleDeg, scale float64, f Filter) (*Gray, error) {
	return RotateScaleWithBound(g, angleDeg, scale, f, DefaultAliasBound)
}

// RotateScaleWithBound 同 RotateScale，但允许显式给定混叠上界 aliasBound；
// aliasBound<=0 时退回 DefaultAliasBound。
func RotateScaleWithBound(g *Gray, angleDeg, scale float64, f Filter, aliasBound float64) (*Gray, error) {
	buf, outW, outH, err := resampleFloat(g, angleDeg, scale, f, aliasBound)
	if err != nil {
		return nil, err
	}
	out, err := New(outW, outH)
	if err != nil {
		return nil, err
	}
	for i, v := range buf {
		q := math.Round(v)
		if q < 0 {
			q = 0
		} else if q > 255 {
			q = 255
		}
		out.Pix[i] = uint8(q)
	}
	return out, nil
}
