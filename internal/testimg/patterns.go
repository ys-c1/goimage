// Package testimg 提供放大核验用的合成图案与降采样工具，
// 供 img 包的测试与 cmd/upscalereport 共用，避免两处用例定义漂移。
package testimg

import (
	"math"

	"goimage/img"
)

// Pattern 描述一张高分辨率参考图及其名称。
type Pattern struct {
	Name string
	Ref  *img.Gray
}

// Size 为参考图边长。
const Size = 128

func build(w, h int, f func(x, y int) float64) *img.Gray {
	g, err := img.New(w, h)
	if err != nil {
		panic(err)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := math.Round(f(x, y))
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			g.Pix[y*w+x] = uint8(v)
		}
	}
	return g
}

// All 返回五类用例：直边、斜边、细线、纹理、纯色。
func All() []Pattern {
	const w, h = Size, Size
	return []Pattern{
		{"直边", build(w, h, func(x, y int) float64 {
			if x < w/2 {
				return 30
			}
			return 220
		})},
		{"斜边", build(w, h, func(x, y int) float64 {
			if float64(y) < 0.6*float64(x)+20 {
				return 40
			}
			return 210
		})},
		{"细线", build(w, h, func(x, y int) float64 {
			if math.Abs(float64(y)-64) < 1.5 {
				return 230
			}
			return 35
		})},
		{"纹理", build(w, h, func(x, y int) float64 {
			return 128 + 70*math.Sin(float64(x)*0.55)*math.Cos(float64(y)*0.47) +
				20*math.Sin(float64(x+y)*1.3)
		})},
		{"纯色", build(w, h, func(x, y int) float64 {
			return 128
		})},
	}
}

// Downsample 对参考图做 scale×scale 盒式平均降采样，得到放大算法的输入。
func Downsample(ref *img.Gray, scale int) *img.Gray {
	ow, oh := ref.W/scale, ref.H/scale
	out, err := img.New(ow, oh)
	if err != nil {
		panic(err)
	}
	for y := 0; y < oh; y++ {
		for x := 0; x < ow; x++ {
			var sum float64
			for dy := 0; dy < scale; dy++ {
				for dx := 0; dx < scale; dx++ {
					sum += float64(ref.Pix[(y*scale+dy)*ref.W+x*scale+dx])
				}
			}
			out.Pix[y*ow+x] = uint8(math.Round(sum / float64(scale*scale)))
		}
	}
	return out
}
