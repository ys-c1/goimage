// histmatch 演示最优直方图匹配，并对照均匀分箱、线性映射两种基线。
package main

import (
	"fmt"
	"math"

	"goimage/img"
)

const levels = 256

func target() []float64 {
	t := make([]float64, levels)
	for v := 0; v < levels; v++ {
		x := float64(v)
		t[v] = 0.7*math.Exp(-x/25) + 0.3*math.Exp(-(x-220)*(x-220)/(2*15*15))
	}
	return t
}

func gauss(h *[levels]float64, mu, sigma, w float64) {
	for v := 0; v < levels; v++ {
		d := (float64(v) - mu) / sigma
		h[v] += w * math.Exp(-d*d/2)
	}
}

func makeImage(h *[levels]float64, n int) *img.Gray {
	g, err := img.New(n, 1)
	if err != nil {
		panic(err)
	}
	sum := 0.0
	for _, v := range h {
		sum += v
	}
	idx := 0
	for v := 0; v < levels; v++ {
		for c := int(math.Round(h[v] / sum * float64(n))); c > 0 && idx < n; c-- {
			g.Pix[idx] = uint8(v)
			idx++
		}
	}
	for idx < n {
		g.Pix[idx] = levels - 1
		idx++
	}
	return g
}

func main() {
	const bins = 64
	tg := target()

	var uni, bi, skew [levels]float64
	gauss(&uni, 80, 15, 1)
	gauss(&bi, 50, 10, 0.5)
	gauss(&bi, 200, 12, 0.5)
	skew[5] = 0.95
	for v := 100; v < levels; v++ {
		skew[v] = 0.05 / 156
	}
	var flat [levels]float64
	for v := range flat {
		flat[v] = 1
	}
	one, _ := img.New(1, 1)
	one.Pix[0] = 150

	cases := []struct {
		name string
		g    *img.Gray
	}{
		{"单峰", makeImage(&uni, 8192)},
		{"双峰", makeImage(&bi, 8192)},
		{"均匀", makeImage(&flat, 8192)},
		{"极端偏斜", makeImage(&skew, 8192)},
		{"单像素", one},
	}

	fmt.Printf("目标分布: 0.7·exp(-v/25) + 0.3·N(220,15²), 分箱数 %d\n", bins)
	fmt.Printf("%-8s %6s %6s %10s %10s %12s %12s\n", "用例", "迭代", "收敛", "W1", "W1上界", "均匀分箱W1", "线性映射W1")
	for _, c := range cases {
		r, err := img.MatchHistogram(c.g, tg, bins)
		if err != nil {
			fmt.Println(c.name, "错误:", err)
			continue
		}
		fmt.Printf("%-8s %6d %6v %10.4f %10.4f %12.4f %12.4f\n",
			c.name, r.Iterations, r.Converged, r.W1, r.W1Bound, r.UniformW1, r.LinearW1)
	}
}
