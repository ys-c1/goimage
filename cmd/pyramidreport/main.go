// pyramidreport 跑五类用例（平滑、纹理、边缘、噪点、纯色），打印每种用例
// 反解层数与各层的实测/上界 MSE 曲线，作为拉普拉斯金字塔重建的收尾核验。
package main

import (
	"fmt"
	"math"
	"math/rand"

	"goimage/img"
)

type gen func(w, h int) *img.Gray

func smooth(w, h int) *img.Gray {
	g, _ := img.New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 128 + 80*math.Sin(2*math.Pi*float64(x)/float64(w))*
				math.Cos(2*math.Pi*float64(y)/float64(h))
			g.Pix[y*w+x] = uint8(math.Round(v))
		}
	}
	return g
}

func texture(w, h int) *img.Gray {
	g, _ := img.New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g.Pix[y*w+x] = uint8(30 + ((x+y)%2)*90 + ((x*3+y*5)%7)*12)
		}
	}
	return g
}

func edge(w, h int) *img.Gray {
	g, _ := img.New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(20)
			if x >= w/2 {
				v = 220
			}
			if y >= h/2 {
				v = 255 - v
			}
			g.Pix[y*w+x] = v
		}
	}
	return g
}

func noise(w, h int) *img.Gray {
	r := rand.New(rand.NewSource(20261124))
	g, _ := img.New(w, h)
	for i := range g.Pix {
		g.Pix[i] = uint8(r.Intn(256))
	}
	return g
}

func solid(w, h int) *img.Gray {
	g, _ := img.New(w, h)
	for i := range g.Pix {
		g.Pix[i] = 137
	}
	return g
}

func main() {
	const (
		w, h      = 128, 96
		targetMSE = 100.0
	)
	k, err := img.BinomialKernel(img.DefaultKernelLen)
	if err != nil {
		panic(err)
	}
	cases := []struct {
		name string
		gen  gen
	}{
		{"平滑", smooth}, {"纹理", texture}, {"边缘", edge},
		{"噪点", noise}, {"纯色", solid},
	}

	fmt.Printf("图幅 %d×%d，核 binomial-%d，目标 MSE=%g\n\n", w, h, img.DefaultKernelLen, targetMSE)
	fmt.Println("## 反解层数与完整重建误差")
	fmt.Println()
	fmt.Println("| 用例 | 几何上限 | 反解层数 | 完整重建 MSE |")
	fmt.Println("| --- | ---: | ---: | ---: |")
	for _, c := range cases {
		g := c.gen(w, h)
		maxLv := img.MaxLevels(w, h)
		n, err := img.SolveLevels(g, targetMSE, k)
		if err != nil {
			fmt.Printf("| %s | %d | 错误: %v | |\n", c.name, maxLv, err)
			continue
		}
		lp, err := img.BuildLaplacian(g, maxLv, k)
		if err != nil {
			panic(err)
		}
		_, measured, _ := lp.ReconstructCurve()
		fmt.Printf("| %s | %d | %d | %.6g |\n", c.name, maxLv, n, measured[maxLv])
	}

	fmt.Println()
	fmt.Println("## 层数—误差曲线（纹理用例，满层塔）")
	fmt.Println()
	fmt.Println("| 已加回层数 | 实测 MSE | 上界 MSE | 实测/上界 |")
	fmt.Println("| ---: | ---: | ---: | ---: |")
	g := texture(w, h)
	maxLv := img.MaxLevels(w, h)
	lp, _ := img.BuildLaplacian(g, maxLv, k)
	_, measured, bounds := lp.ReconstructCurve()
	for l := 0; l <= maxLv; l++ {
		ratio := 1.0
		if bounds[l] > 0 {
			ratio = measured[l] / bounds[l]
		}
		fmt.Printf("| %d | %.4f | %.4f | %.6f |\n", l, measured[l], bounds[l], ratio)
	}

	fmt.Println()
	fmt.Println("## 输入校验（三类非法输入各报参数名）")
	if _, err := img.BuildLaplacian(g, 0, k); err != nil {
		fmt.Println("层数为零 →", err)
	}
	if _, err := img.GaussianKernel(4, 1); err != nil {
		fmt.Println("核长为偶 →", err)
	}
	if _, err := img.BuildLaplacian(nil, 2, k); err != nil {
		fmt.Println("图为空   →", err)
	}
}
