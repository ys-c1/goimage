package img

import (
	"errors"
	"fmt"
	"math"
)

// RGB 是一组编码（非线性）RGB 值，各通道约定在 [0,1]。
type RGB struct {
	R, G, B float64
}

// ColorImage 是一幅编码 RGB 图像，Pix 长度必须等于 W*H。
type ColorImage struct {
	W, H int
	Pix  []RGB
}

// NewColorImage 创建一幅全黑彩色图像。
func NewColorImage(w, h int) (*ColorImage, error) {
	if w <= 0 || h <= 0 {
		return nil, ErrSize
	}
	return &ColorImage{W: w, H: h, Pix: make([]RGB, w*h)}, nil
}

var (
	// ErrGamma 表示 gamma 不是有限正数。
	ErrGamma = errors.New("gamma 非法")
	// ErrSingular 表示色域矩阵不可逆（行列式为零或元素非有限）。
	ErrSingular = errors.New("矩阵奇异")
	// ErrEmptyColor 表示彩色图为空或尺寸与像素数不一致。
	ErrEmptyColor = errors.New("图为空")
	// ErrKnee 表示软拐点比例不在 (0,1)。
	ErrKnee = errors.New("拐点比例必须落在 (0,1)")
)

// FieldError 把出错字段名与具体错误绑定，调用方可用 errors.As 取出字段。
type FieldError struct {
	Field string
	Err   error
}

func (e *FieldError) Error() string { return fmt.Sprintf("%s: %v", e.Field, e.Err) }
func (e *FieldError) Unwrap() error { return e.Err }

// Mat3 是行主序 3x3 矩阵。
type Mat3 [3][3]float64

// MulVec 计算 m*v。
func (m Mat3) MulVec(v [3]float64) [3]float64 {
	return [3]float64{
		m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
		m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
		m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
	}
}

// Mul 计算 m*n。
func (m Mat3) Mul(n Mat3) Mat3 {
	var out Mat3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = m[i][0]*n[0][j] + m[i][1]*n[1][j] + m[i][2]*n[2][j]
		}
	}
	return out
}

// Det 返回行列式。
func (m Mat3) Det() float64 {
	return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}

// Inv 返回逆矩阵；行列式接近零或结果非有限时返回 ErrSingular。
func (m Mat3) Inv() (Mat3, error) {
	det := m.Det()
	if math.IsNaN(det) || math.IsInf(det, 0) || math.Abs(det) < 1e-12 {
		return Mat3{}, ErrSingular
	}
	inv := Mat3{
		{m[1][1]*m[2][2] - m[1][2]*m[2][1], m[0][2]*m[2][1] - m[0][1]*m[2][2], m[0][1]*m[1][2] - m[0][2]*m[1][1]},
		{m[1][2]*m[2][0] - m[1][0]*m[2][2], m[0][0]*m[2][2] - m[0][2]*m[2][0], m[0][2]*m[1][0] - m[0][0]*m[1][2]},
		{m[1][0]*m[2][1] - m[1][1]*m[2][0], m[0][1]*m[2][0] - m[0][0]*m[2][1], m[0][0]*m[1][1] - m[0][1]*m[1][0]},
	}
	for i := range inv {
		for j := range inv[i] {
			inv[i][j] /= det
			if math.IsNaN(inv[i][j]) || math.IsInf(inv[i][j], 0) {
				return Mat3{}, ErrSingular
			}
		}
	}
	return inv, nil
}

// 矩阵由原色坐标直接推导（D65 白点），避免不同来源公开矩阵之间的舍入不一致。
// 推导方式：以各原色 xyz 为列构成 P，解 P*S=W 得白点缩放，M=P*diag(S)。
var (
	sRGBToXYZD65Matrix = Mat3{
		{0.41239079926595934, 0.35758433938387796, 0.18048078840183430},
		{0.21263900587151027, 0.71516867876775590, 0.07219231536073371},
		{0.01933081871559185, 0.11919477979462599, 0.95053215224966070},
	}
	displayP3ToXYZD65Matrix = Mat3{
		{0.48657094864821620, 0.26566769316909306, 0.19821728523436250},
		{0.22897456406974880, 0.69173852183650640, 0.07928691409374500},
		{0.00000000000000000, 0.04511338185890264, 1.04394436890097600},
	}
)

// SRGBToXYZD65 返回线性 sRGB 到 D65 XYZ 的矩阵。
func SRGBToXYZD65() Mat3 { return sRGBToXYZD65Matrix }

// DisplayP3ToXYZD65 返回线性 Display P3 到 D65 XYZ 的矩阵。
func DisplayP3ToXYZD65() Mat3 { return displayP3ToXYZD65Matrix }

// decodeGamma 把编码值解为线性光。输入先截到 [0,1]。
func decodeGamma(v, gamma float64) float64 {
	if !(v > 0) {
		return 0
	}
	if v >= 1 {
		return 1
	}
	return math.Pow(v, gamma)
}

// encodeGamma 把线性光编码为码值。输入先截到 [0,1]。
func encodeGamma(v, gamma float64) float64 {
	if !(v > 0) {
		return 0
	}
	if v >= 1 {
		return 1
	}
	return math.Pow(v, 1/gamma)
}

func decodeRGB(c RGB, gamma float64) [3]float64 {
	return [3]float64{decodeGamma(c.R, gamma), decodeGamma(c.G, gamma), decodeGamma(c.B, gamma)}
}

func encodeRGB(v [3]float64, gamma float64) RGB {
	return RGB{
		R: encodeGamma(v[0], gamma),
		G: encodeGamma(v[1], gamma),
		B: encodeGamma(v[2], gamma),
	}
}

// XYZToOklab 把 D65 XYZ 转到 Oklab（感知均匀空间）。
func XYZToOklab(xyz [3]float64) [3]float64 {
	l := 0.8189330101*xyz[0] + 0.3618667424*xyz[1] - 0.1288597137*xyz[2]
	m := 0.0329845436*xyz[0] + 0.9293118715*xyz[1] + 0.0361456387*xyz[2]
	s := 0.0482003018*xyz[0] + 0.2643662691*xyz[1] + 0.6338517070*xyz[2]
	l, m, s = math.Cbrt(l), math.Cbrt(m), math.Cbrt(s)
	return [3]float64{
		0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
		1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
		0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
	}
}

// OklabToXYZ 把 Oklab 转回 D65 XYZ。
func OklabToXYZ(lab [3]float64) [3]float64 {
	L, a, b := lab[0], lab[1], lab[2]
	l := L + 0.3963377774*a + 0.2158037573*b
	m := L - 0.1055613458*a - 0.0638541728*b
	s := L - 0.0894841775*a - 1.2914855480*b
	l, m, s = l*l*l, m*m*m, s*s*s
	return [3]float64{
		1.2270138511*l - 0.5577999807*m + 0.2812561490*s,
		-0.0405801784*l + 1.1122568696*m - 0.0716766787*s,
		-0.0763812845*l - 0.4214819784*m + 1.5861632204*s,
	}
}

// hueShiftDeg 返回 src 到 out 在 Oklab a-b 平面上的带符号色相旋转角（度）。
// 任一端彩度接近零时色相无定义，返回 0。
func hueShiftDeg(srcLab, outLab [3]float64) float64 {
	cs := math.Hypot(srcLab[1], srcLab[2])
	co := math.Hypot(outLab[1], outLab[2])
	if cs < 1e-6 || co < 1e-6 {
		return 0
	}
	cross := srcLab[1]*outLab[2] - srcLab[2]*outLab[1]
	dot := srcLab[1]*outLab[1] + srcLab[2]*outLab[2]
	return math.Atan2(cross, dot) * 180 / math.Pi
}

// deltaEOK 是 Oklab 中的欧氏色差。
func deltaEOK(a, b [3]float64) float64 {
	dl := a[0] - b[0]
	da := a[1] - b[1]
	db := a[2] - b[2]
	return math.Sqrt(dl*dl + da*da + db*db)
}
