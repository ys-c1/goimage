package img

import (
	"math"
)

// GamutConfig 描述一次“源色域编码值 -> 目标色域编码值”的映射。
type GamutConfig struct {
	// Gamma 是源与目标共用的编码幂指数（例如 sRGB/P3 近似为 2.2）。
	Gamma float64
	// SourceToXYZ 把源线性 RGB 转到 D65 XYZ。
	SourceToXYZ Mat3
	// TargetToXYZ 把目标线性 RGB 转到 D65 XYZ。
	TargetToXYZ Mat3
	// Knee 是软压缩起点占目标边界彩度的比例，必须在 (0,1)；为零时取默认 0.9。
	Knee float64
}

// DefaultGamutConfig 以 Display P3 为源、sRGB 为目标，gamma 取 2.2。
func DefaultGamutConfig() GamutConfig {
	return GamutConfig{
		Gamma:       2.2,
		SourceToXYZ: DisplayP3ToXYZD65(),
		TargetToXYZ: SRGBToXYZD65(),
		Knee:        0.9,
	}
}

const defaultKnee = 0.9

func (c GamutConfig) knee() float64 {
	if c.Knee == 0 {
		return defaultKnee
	}
	return c.Knee
}

func matrixFinite(m Mat3) bool {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.IsNaN(m[i][j]) || math.IsInf(m[i][j], 0) {
				return false
			}
		}
	}
	return true
}

func (c GamutConfig) validate() error {
	if !(c.Gamma > 0) || math.IsNaN(c.Gamma) || math.IsInf(c.Gamma, 0) {
		return &FieldError{Field: "gamma", Err: ErrGamma}
	}
	if !matrixFinite(c.SourceToXYZ) {
		return &FieldError{Field: "sourceMatrix", Err: ErrSingular}
	}
	if _, err := c.SourceToXYZ.Inv(); err != nil {
		return &FieldError{Field: "sourceMatrix", Err: ErrSingular}
	}
	if !matrixFinite(c.TargetToXYZ) {
		return &FieldError{Field: "targetMatrix", Err: ErrSingular}
	}
	if _, err := c.TargetToXYZ.Inv(); err != nil {
		return &FieldError{Field: "targetMatrix", Err: ErrSingular}
	}
	k := c.knee()
	if !(k > 0) || k >= 1 || math.IsNaN(k) {
		return &FieldError{Field: "knee", Err: ErrKnee}
	}
	return nil
}

func validateColorImage(im *ColorImage) error {
	if im == nil || im.W <= 0 || im.H <= 0 || len(im.Pix) == 0 || len(im.Pix) != im.W*im.H {
		return &FieldError{Field: "image", Err: ErrEmptyColor}
	}
	return nil
}

// 映射器内部使用线性 RGB、XYZ、Oklab 三种表示，矩阵方向固定：
// 源线性 -> XYZ -> Oklab ->（限幅）-> XYZ -> 目标线性 -> 编码。
type gamutMapper struct {
	cfg       GamutConfig
	targetInv Mat3
}

func newGamutMapper(cfg GamutConfig) (*gamutMapper, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	inv, err := cfg.TargetToXYZ.Inv()
	if err != nil {
		return nil, &FieldError{Field: "targetMatrix", Err: ErrSingular}
	}
	return &gamutMapper{cfg: cfg, targetInv: inv}, nil
}

// inGamut 判断目标线性 RGB 是否落在 [0,1] 内（含边界容差）。
// 容差只用于吸收浮点误差，不参与限制逻辑。NaN 一律视为域外。
func inGamut(rgb [3]float64) bool {
	const tol = 1e-9
	for _, v := range rgb {
		if !(v >= -tol && v <= 1+tol) {
			return false
		}
	}
	return true
}

// targetRGB 把 Oklab 颜色转成目标线性 RGB。
func (m *gamutMapper) targetRGB(lab [3]float64) [3]float64 {
	return m.targetInv.MulVec(OklabToXYZ(lab))
}

// boundaryChroma 返回亮度 L、色相 h 上目标色域边界处的 Oklab 彩度。
// 中性轴本身已在色域外（例如 L 超出范围）时返回零。
func (m *gamutMapper) boundaryChroma(L, h float64) float64 {
	if L >= 1-1e-9 {
		return 0
	}
	if !inGamut(m.targetRGB([3]float64{L, 0, 0})) {
		return 0
	}
	// 先以倍增方式框住边界，再二分到浮点精度。
	lo, hi := 0.0, 1e-12
	for hi < 8 && inGamut(m.targetRGB(oklchToOklab(L, hi, h))) {
		lo = hi
		hi *= 2
	}
	for i := 0; i < 52; i++ {
		mid := (lo + hi) / 2
		if inGamut(m.targetRGB(oklchToOklab(L, mid, h))) {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo
}

// softClipChroma 是自推导的压缩曲线，把 C 从 [0, ∞) 映到 [0, Cmax]。
//
// 推导：设 C0=Knee*Cmax，A=Cmax-C0。
//   - C <= C0 时恒等，这段只出现在“源色本就在目标域内”的旁路上；
//   - C >  C0 时取 Cp = Cmax - A*exp(-(C-C0)/A)。
//
// 该式由三个约束唯一确定：
//  1. Cp(C0)=Cmax-A=C0，在拐点处与恒等段连续；
//  2. Cp'(C)=exp(-(C-C0)/A)，故 Cp'(C0)=1，一阶导数也连续，曲线无折角，
//     不会在拐点附近把平滑渐变压出可见的分层；
//  3. C->∞ 时 Cp->Cmax 且始终小于 Cmax，收敛到边界而不越界。
//
// 指数衰减的斜率随超出量单调下降（Cp”= -(1/A)exp(...) < 0），越远的颜色被压得越多，
// 相比直接截断（C>=Cmax 全部变 Cmax）保留了超域颜色之间的彩度差异，
// 因此饱和红、饱和绿不会塌成同一个纯色块。
func softClipChroma(C, Cmax, knee float64) float64 {
	if Cmax <= 0 {
		return 0
	}
	c0 := knee * Cmax
	a := Cmax - c0
	if C <= c0 {
		return C
	}
	return Cmax - a*math.Exp(-(C-c0)/a)
}

func oklchToOklab(L, C, h float64) [3]float64 {
	return [3]float64{L, C * math.Cos(h), C * math.Sin(h)}
}

// mapLab 对单个 Oklab 颜色做色域映射，返回目标线性 RGB。
func (m *gamutMapper) mapLab(srcLab [3]float64) [3]float64 {
	L, C, h := srcLab[0], math.Hypot(srcLab[1], srcLab[2]), math.Atan2(srcLab[2], srcLab[1])
	Lc := min(max(L, 0), 1)
	Cmax := m.boundaryChroma(Lc, h)
	Cp := softClipChroma(C, Cmax, m.cfg.knee())
	return m.targetRGB(oklchToOklab(Lc, Cp, h))
}

// MapPixel 按配置映射单个编码 RGB 值，返回目标色域编码值。
func MapPixel(src RGB, cfg GamutConfig) (RGB, error) {
	m, err := newGamutMapper(cfg)
	if err != nil {
		return RGB{}, err
	}
	return m.mapEncoded(src), nil
}

// mapEncodedFull 返回映射后的目标编码值，以及“编码前目标线性值是否仍越界”。
// 后者是独立的残差信号：即便编码把值截到 [0,1]，只要截断真的发生了就会报告 true，
// 不会被最终的 clamp 掩盖。
func (m *gamutMapper) mapEncodedFull(src RGB) (RGB, bool) {
	lin := decodeRGB(src, m.cfg.Gamma)
	xyz := m.cfg.SourceToXYZ.MulVec(lin)
	targetLin := m.targetInv.MulVec(xyz)

	// 已在目标域内的颜色只做空间转换，不做任何限幅。
	if inGamut(targetLin) {
		return encodeRGB(targetLin, m.cfg.Gamma), false
	}
	mapped := m.mapLab(XYZToOklab(xyz))
	return encodeRGB(mapped, m.cfg.Gamma), !inGamut(mapped)
}

func (m *gamutMapper) mapEncoded(src RGB) RGB {
	out, _ := m.mapEncodedFull(src)
	return out
}

// MapToGamut 把整幅彩色图从源色域映射到目标色域，返回新图，不修改输入。
func MapToGamut(src *ColorImage, cfg GamutConfig) (*ColorImage, error) {
	if err := validateColorImage(src); err != nil {
		return nil, err
	}
	m, err := newGamutMapper(cfg)
	if err != nil {
		return nil, err
	}
	out, err := NewColorImage(src.W, src.H)
	if err != nil {
		return nil, err
	}
	for i, c := range src.Pix {
		out.Pix[i] = m.mapEncoded(c)
	}
	return out, nil
}

// 以下诊断函数都把编码值重新解码，与映射实际走同一条管线。

// linearTargetOf 返回编码值经源矩阵转到目标线性 RGB 的结果（不做映射）。
func (m *gamutMapper) linearTargetOf(src RGB) [3]float64 {
	return m.targetInv.MulVec(m.cfg.SourceToXYZ.MulVec(decodeRGB(src, m.cfg.Gamma)))
}

// outOfGamut 判断源编码值在目标色域中是否超域（比较编码前的线性值）。
func (m *gamutMapper) outOfGamut(src RGB) bool {
	return !inGamut(m.linearTargetOf(src))
}

// labOfTarget 把目标编码值转回 Oklab，用于色差计算。
func (m *gamutMapper) labOfTarget(c RGB) [3]float64 {
	lin := decodeRGB(c, m.cfg.Gamma)
	return XYZToOklab(m.cfg.TargetToXYZ.MulVec(lin))
}

// labOfSource 把源编码值转到 Oklab。
func (m *gamutMapper) labOfSource(src RGB) [3]float64 {
	lin := decodeRGB(src, m.cfg.Gamma)
	return XYZToOklab(m.cfg.SourceToXYZ.MulVec(lin))
}
