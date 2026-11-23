// ditherreport 量化对照报告：逐像素量化参考图 vs 蛇形误差扩散抖动。
//
// 用法示例：
//
//	go run ./cmd/ditherreport
//	go run ./cmd/ditherreport -kernel stucki -bits 3 -strength 1
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"goimage/img"
)

func gradient(w, h int) *img.Gray {
	g, _ := img.New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g.Pix[y*w+x] = uint8(x * 255 / (w - 1))
		}
	}
	return g
}

func pct(before, after int) string {
	r, ok := img.BandReduction(before, after)
	if !ok {
		return "无参考基线"
	}
	return fmt.Sprintf("%+.1f%%", 100*r)
}

func main() {
	name := flag.String("kernel", "floyd-steinberg", "扩散核名字（见下表或 -list）")
	bits := flag.Int("bits", 2, "目标位深 1..8")
	strength := flag.Float64("strength", 1, "扩散强度，非负")
	list := flag.Bool("list", false, "只列出内置扩散核")
	flag.Parse()

	if *list {
		for _, k := range img.BuiltinKernels() {
			fmt.Println(k.Name)
		}
		return
	}

	k, err := img.LookupKernel(*name)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Printf("扩散核 %s：逐项权重表\n", k.Name)
	fmt.Printf("  %-4s %-12s %-10s %s\n", "序号", "偏移(dx,dy)", "权重", "分数形式")
	sum := 0.0
	for i := range k.W {
		fmt.Printf("  %-6d %-14s %-12.6f %s\n",
			i, fmt.Sprintf("(%+d,%+d)", k.DX[i], k.DY[i]), k.W[i], fraction(k.W[i]))
		sum += k.W[i]
	}
	fmt.Printf("  共 %d 项，权重和 %.6f\n\n", len(k.W), sum)

	levels := 1 << *bits
	fmt.Printf("位深 %d（%d 色阶），强度 %g\n\n", *bits, levels, *strength)
	header := fmt.Sprintf("%-10s %-10s %-8s %-8s %-10s %-10s %-8s", "图幅", "口径", "参考图", "抖动后", "降幅", "MAE(原图)", "耗时")
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)+8))
	for _, sz := range [][2]int{{256, 256}, {1024, 1024}} {
		w, h := sz[0], sz[1]
		src := gradient(w, h)

		t0 := time.Now()
		ref, err := img.Quantize(src, levels)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		tRef := time.Since(t0)

		t0 = time.Now()
		out, err := img.Dither(src, img.DitherOptions{Bits: *bits, Strength: *strength, Kernel: k})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		tOut := time.Since(t0)
		t0 = time.Now()
		refBands := img.CountBands(ref)
		tBands := time.Since(t0)
		outBands := img.CountBands(out)

		mae, _ := img.MeanAbsError(out, src)

		label := fmt.Sprintf("%dx%d", w, h)
		fmt.Printf("%-10s %-10s %-8d %-8d %-10s %-10.3f %-8s\n",
			label, "A 整线", refBands.ScanLines, outBands.ScanLines,
			pct(refBands.ScanLines, outBands.ScanLines), mae, tOut.Round(time.Microsecond))
		fmt.Printf("%-10s %-10s %-8d %-8d %-10s %-10s %-8s\n",
			"", "B 长游程", refBands.LongRuns, outBands.LongRuns,
			pct(refBands.LongRuns, outBands.LongRuns), "", "")
		fmt.Printf("%-10s %-10s %-8s %-8s %-10s %-10s %-8s\n",
			"", "参考图耗时", "", "",
			"", "", tRef.Round(time.Microsecond))
		fmt.Printf("%-10s %-10s %-8s %-8s %-10s %-10s %-8s\n",
			"", "环带统计耗时", "", "",
			"", "", tBands.Round(time.Microsecond))
	}
}

func fraction(w float64) string {
	for _, d := range []int{4, 8, 16, 32, 42, 48} {
		n := w * float64(d)
		if n == float64(int(n+0.5)) && abs(n-float64(int(n+0.5))) < 1e-9 {
			return fmt.Sprintf("%d/%d", int(n+0.5), d)
		}
	}
	return ""
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
