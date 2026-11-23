package demo

import (
	"math"

	"goimage/img"
)

// MSE 返回均方误差；两图尺寸必须一致。
func MSE(a, b *img.Gray) float64 {
	if a.W != b.W || a.H != b.H {
		panic("MSE: 尺寸不一致")
	}
	if len(a.Pix) == 0 {
		return 0
	}
	s := 0.0
	for i := range a.Pix {
		d := float64(a.Pix[i]) - float64(b.Pix[i])
		s += d * d
	}
	return s / float64(len(a.Pix))
}

// PSNR 由 MSE 换算峰值信噪比；MSE 为零时返回 +Inf。
func PSNR(mse float64) float64 {
	if mse == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(255*255/mse)
}

// MaxErr 返回逐像素最大绝对误差。
func MaxErr(a, b *img.Gray) int {
	if a.W != b.W || a.H != b.H {
		panic("MaxErr: 尺寸不一致")
	}
	m := 0
	for i := range a.Pix {
		d := int(a.Pix[i]) - int(b.Pix[i])
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}
