package img

import (
	"errors"
	"math"
)

var (
	// ErrParamLevels 层数为零时返回，并指出参数名 levels。
	ErrParamLevels = errors.New("参数 levels 非法：层数必须为正")
	// ErrParamKernel 核长为偶（或非正）时返回，并指出参数名 kernelLen。
	ErrParamKernel = errors.New("参数 kernelLen 非法：高斯核长度必须为正奇数")
	// ErrParamImage 图为空（nil 或零画幅）时返回，并指出参数名 image。
	ErrParamImage = errors.New("参数 image 非法：图为空")
	// ErrParamSigma 高斯核标准差非正时返回，并指出参数名 sigma。
	ErrParamSigma = errors.New("参数 sigma 非法：标准差必须为正")
	// ErrParamTarget 目标画幅非正时返回，并指出参数名 width/height。
	ErrParamTarget = errors.New("参数 width/height 非法：目标画幅必须为正")
	// ErrParamTol 目标误差非正时返回，并指出参数名 targetMSE。
	ErrParamTol = errors.New("参数 targetMSE 非法：目标误差必须为正")
	// ErrLevelsTooDeep 层数超过该图幅几何允许的最大降采样层数。
	ErrLevelsTooDeep = errors.New("参数 levels 非法：层数超过图幅允许的最大降采样层数")
	// ErrTolUnreachable 图已降到最粗层仍无法达到目标误差。
	ErrTolUnreachable = errors.New("目标误差在最大层数下仍不可达")
)

// clamp 把坐标钳制到 [0,n-1]，即 replicate 边界。
func clamp(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// toFloat 把灰度图转成 [0,255] 浮点工作图。
func toFloat(g *Gray) *fimg {
	f := &fimg{W: g.W, H: g.H, Pix: make([]float64, len(g.Pix))}
	for i, v := range g.Pix {
		f.Pix[i] = float64(v)
	}
	return f
}

// toGray 把浮点工作图转回 uint8 灰度图，结果钳制到 [0,255] 并四舍五入。
func (f *fimg) toGray() *Gray {
	g := &Gray{W: f.W, H: f.H, Pix: make([]uint8, len(f.Pix))}
	for i, v := range f.Pix {
		v = math.Round(v)
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		g.Pix[i] = uint8(v)
	}
	return g
}

// resizeFloat 是单层双线性重采样：源坐标按画幅比中心对齐映射，
// 边界采用 replicate 钳制。仅做一层插值，不含任何金字塔细节增强。
func resizeFloat(src *fimg, dw, dh int) *fimg {
	dst := &fimg{W: dw, H: dh, Pix: make([]float64, dw*dh)}
	if src.W == 1 {
		// 宽度为 1 时无横向插值空间，直接列复制。
		for dy := 0; dy < dh; dy++ {
			sy := clamp(int((float64(dy)+0.5)*float64(src.H)/float64(dh)-0.5), src.H)
			base := sy * src.W
			row := dy * dw
			for dx := 0; dx < dw; dx++ {
				dst.Pix[row+dx] = src.Pix[base]
			}
		}
		return dst
	}
	for dy := 0; dy < dh; dy++ {
		sy := (float64(dy)+0.5)*float64(src.H)/float64(dh) - 0.5
		y0 := int(math.Floor(sy))
		fy := sy - float64(y0)
		ya, yb := clamp(y0, src.H), clamp(y0+1, src.H)
		row := dy * dw
		for dx := 0; dx < dw; dx++ {
			sx := (float64(dx)+0.5)*float64(src.W)/float64(dw) - 0.5
			x0 := int(math.Floor(sx))
			fx := sx - float64(x0)
			xa, xb := clamp(x0, src.W), clamp(x0+1, src.W)
			v00 := src.Pix[ya*src.W+xa]
			v01 := src.Pix[ya*src.W+xb]
			v10 := src.Pix[yb*src.W+xa]
			v11 := src.Pix[yb*src.W+xb]
			top := v00*(1-fx) + v01*fx
			bot := v10*(1-fx) + v11*fx
			dst.Pix[row+dx] = top*(1-fy) + bot*fy
		}
	}
	return dst
}

// Resize 是原有的单层缩放接口：一层双线性插值，不做金字塔重建。
// 该接口的行为保持不变，金字塔增强由 [ResizePyramid] 提供。
func Resize(g *Gray, width, height int) (*Gray, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return nil, ErrParamImage
	}
	if width <= 0 || height <= 0 {
		return nil, ErrParamTarget
	}
	if width == g.W && height == g.H {
		out := &Gray{W: g.W, H: g.H, Pix: append([]uint8(nil), g.Pix...)}
		return out, nil
	}
	return resizeFloat(toFloat(g), width, height).toGray(), nil
}
