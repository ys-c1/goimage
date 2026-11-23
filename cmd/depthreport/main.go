// depthreport 跑五个用例 × 两档图幅，打印色阶间距表与扩散强度对应表。
package main

import (
	"fmt"
	"math"
	"math/rand"

	"goimage/img"
)

type gen func(w, h int) *img.Gray16

func gradient(w, h int) *img.Gray16 {
	g, _ := img.New16(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g.Pix[y*w+x] = uint16(uint32(x) * 65535 / uint32(w-1))
		}
	}
	return g
}

func photo(w, h int) *img.Gray16 {
	r := rand.New(rand.NewSource(42))
	g, _ := img.New16(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 0.5 + 0.3*math.Sin(float64(x)*0.05)*math.Cos(float64(y)*0.04) + 0.15*(r.Float64()-0.5)
			if v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			g.Pix[y*w+x] = uint16(math.Round(v * 65535))
		}
	}
	return g
}

func icon(w, h int) *img.Gray16 {
	g, _ := img.New16(w, h)
	cx, cy, rad := float64(w)/2, float64(h)/2, float64(w)/4
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint16(6553)
			dx, dy := float64(x)-cx, float64(y)-cy
			if dx*dx+dy*dy < rad*rad {
				v = 58981
			}
			if x > w/2 && y > h/2 {
				v = 36044
			}
			g.Pix[y*w+x] = v
		}
	}
	return g
}

func solid(w, h int) *img.Gray16 {
	g, _ := img.New16(w, h)
	for i := range g.Pix {
		g.Pix[i] = 30000
	}
	return g
}

func single(w, h int) *img.Gray16 {
	g, _ := img.New16(1, 1)
	g.Pix[0] = 40000
	return g
}

func main() {
	opts := img.CompressOptions{
		SrcDepth:   16,
		DstDepth:   8,
		LoQuantile: 0.01,
		HiQuantile: 0.99,
		Levels:     8,
		MinSpacing: 12,
		MaxSpacing: 24,
		Kernel:     img.KernelFloydSteinberg,
	}
	sizes := []struct {
		name string
		w, h int
	}{{"小图 128×128", 128, 128}, {"大图 1024×1024", 1024, 1024}}
	cases := []struct {
		name string
		gen  gen
	}{
		{"渐变", gradient},
		{"照片", photo},
		{"图标", icon},
		{"纯色", solid},
		{"单像素", single},
	}

	fmt.Println("## 色阶间距表")
	fmt.Println()
	fmt.Println("| 用例 | 图幅 | 色阶数 | 相邻色阶间距 | 耗时 |")
	fmt.Println("| --- | --- | ---: | ---: | ---: |")
	type row struct {
		size     string
		strength float64
		spacing  float64
	}
	corr := map[string]row{}
	for _, c := range cases {
		for _, s := range sizes {
			_, rep, err := img.Compress16(c.gen(s.w, s.h), opts)
			if err != nil {
				fmt.Println("|", c.name, "|", s.name, "| 错误:", err, "| | |")
				continue
			}
			fmt.Printf("| %s | %s | %d | %.2f | %s |\n",
				c.name, s.name, rep.LevelsUsed, rep.Spacing, rep.Elapsed)
			if c.name == "渐变" {
				corr[s.name] = row{s.name, rep.Strength, rep.Spacing}
			}
		}
	}

	fmt.Println()
	fmt.Println("## 扩散强度与色阶间距对应表（以渐变用例为准）")
	fmt.Println()
	fmt.Println("| 图幅 | 扩散强度 α | 色阶间距 |")
	fmt.Println("| --- | ---: | ---: |")
	for _, s := range sizes {
		r := corr[s.name]
		fmt.Printf("| %s | %.4f | %.2f |\n", r.size, r.strength, r.spacing)
	}

	fmt.Println()
	fmt.Println("## 字段名校验")
	bad := opts
	bad.DstDepth = 16
	_, _, err := img.Compress16(gradient(8, 8), bad)
	fmt.Println("位深越界 →", err)
	bad = opts
	bad.LoQuantile, bad.HiQuantile = 0.9, 0.1
	_, _, err = img.Compress16(gradient(8, 8), bad)
	fmt.Println("分位倒置 →", err)
	_, _, err = img.Compress16(&img.Gray16{}, opts)
	fmt.Println("图为空   →", err)
}
