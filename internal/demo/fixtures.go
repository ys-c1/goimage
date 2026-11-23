package demo

import (
	"math"
	"math/rand"

	"goimage/img"
)

// Size 是内置用例的画幅，8 的整数倍。
const Size = 64

// Plain 纯色：所有像素同一灰度。
func Plain() *img.Gray {
	g, _ := img.New(Size, Size)
	for i := range g.Pix {
		g.Pix[i] = 128
	}
	return g
}

// Flat 平坦：只有缓慢的倾斜照明变化，几乎没有细节。
func Flat() *img.Gray {
	g, _ := img.New(Size, Size)
	for y := 0; y < Size; y++ {
		for x := 0; x < Size; x++ {
			g.Pix[y*Size+x] = uint8(96 + x*100/(Size-1) + y*50/(Size-1))
		}
	}
	return g
}

// Texture 纹理：多个空间频率的正弦叠加。
func Texture() *img.Gray {
	g, _ := img.New(Size, Size)
	for y := 0; y < Size; y++ {
		for x := 0; x < Size; x++ {
			v := 128 +
				42*math.Sin(2*math.Pi*float64(x)/8) +
				26*math.Cos(2*math.Pi*float64(y)/16) +
				18*math.Sin(2*math.Pi*float64(x+y)/4)
			g.Pix[y*Size+x] = clamp(v)
		}
	}
	return g
}

// Edges 边缘：斜边与硬阶跃，阶跃位置不对齐 8 的块边界。
func Edges() *img.Gray {
	g, _ := img.New(Size, Size)
	for y := 0; y < Size; y++ {
		for x := 0; x < Size; x++ {
			if x*2+3*y > 3*(Size-1) {
				g.Pix[y*Size+x] = 210
			} else {
				g.Pix[y*Size+x] = 45
			}
		}
	}
	return g
}

// Noise 噪点：固定种子的均匀随机灰度。
func Noise() *img.Gray {
	g, _ := img.New(Size, Size)
	r := rand.New(rand.NewSource(7))
	for i := range g.Pix {
		g.Pix[i] = uint8(r.Intn(256))
	}
	return g
}

// Cases 返回全部用例，名称与生成函数。
func Cases() []struct {
	Name string
	New  func() *img.Gray
} {
	return []struct {
		Name string
		New  func() *img.Gray
	}{
		{"纯色", Plain},
		{"平坦", Flat},
		{"纹理", Texture},
		{"边缘", Edges},
		{"噪点", Noise},
	}
}

func clamp(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(math.Round(v))
}
