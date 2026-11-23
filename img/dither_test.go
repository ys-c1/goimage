package img

import (
	"errors"
	"math"
	"testing"
)

func mustNew(t *testing.T, w, h int) *Gray {
	t.Helper()
	g, err := New(w, h)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func fill(g *Gray, f func(x, y int) uint8) {
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			g.Pix[y*g.W+x] = f(x, y)
		}
	}
}

func gradient(w, h int) *Gray {
	g, _ := New(w, h)
	fill(g, func(x, y int) uint8 { return uint8(x * 255 / (w - 1)) })
	return g
}

func solid(w, h int, v uint8) *Gray {
	g, _ := New(w, h)
	for i := range g.Pix {
		g.Pix[i] = v
	}
	return g
}

func checker(w, h int) *Gray {
	g, _ := New(w, h)
	fill(g, func(x, y int) uint8 {
		if (x+y)%2 == 0 {
			return 0
		}
		return 255
	})
	return g
}

// 细密纹理：确定性伪随机噪声，频率比 2×2 棋盘更高。
func fineTexture(w, h int) *Gray {
	g, _ := New(w, h)
	state := uint32(0x12345678)
	fill(g, func(x, y int) uint8 {
		state = state*1664525 + 1013904223
		v := uint8(state >> 24)
		if (x+y)%3 == 0 {
			v = 255 - v
		}
		return v
	})
	return g
}

func fsKernel(t *testing.T) *Kernel {
	t.Helper()
	k, err := LookupKernel("floyd-steinberg")
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func TestDitherPassthroughIsByteIdentical(t *testing.T) {
	inputs := map[string]*Gray{
		"灰阶":   gradient(37, 23),
		"纯色":   solid(19, 11, 128),
		"细密纹理": fineTexture(31, 17),
		"全黑":   solid(9, 9, 0),
	}
	for name, g := range inputs {
		for _, strength := range []float64{0, 1} {
			out, err := Dither(g, DitherOptions{Bits: 8, Strength: strength, Kernel: fsKernel(t)})
			if err != nil {
				t.Fatalf("%s strength=%v: %v", name, strength, err)
			}
			if out.W != g.W || out.H != g.H || len(out.Pix) != len(g.Pix) {
				t.Fatalf("%s: 画幅改变", name)
			}
			for i := range out.Pix {
				if out.Pix[i] != g.Pix[i] {
					t.Fatalf("%s strength=%v: 直通第 %d 像素被改动（%d→%d）",
						name, strength, i, g.Pix[i], out.Pix[i])
				}
			}
			if out == g {
				t.Fatalf("%s: 直通输出不应与输入共用底层数组", name)
			}
		}
	}
	// 直通不要求扩散核。
	out, err := Dither(gradient(5, 4), DitherOptions{Bits: 8})
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := MeanAbsError(out, gradient(5, 4)); got != 0 {
		t.Fatalf("直通误差应为 0，实际 %v", got)
	}
}

func TestDitherNamedErrors(t *testing.T) {
	g := solid(8, 8, 100)
	good := fsKernel(t)

	t.Run("核参数非法", func(t *testing.T) {
		bad := []struct {
			name string
			k    *Kernel
		}{
			{"空核", nil},
			{"匿名核", &Kernel{W: []float64{1}}},
			{"权重表为空", &Kernel{Name: "empty"}},
			{"偏移缺项", &Kernel{Name: "缺项", DX: []int{1, 0}, DY: []int{0}, W: []float64{1, 2}}},
			{"权重缺项", &Kernel{Name: "缺项2", DX: []int{1}, DY: []int{0}, W: []float64{}}},
			{"负权重", &Kernel{Name: "neg", DX: []int{1}, DY: []int{0}, W: []float64{-0.5}}},
			{"零权重", &Kernel{Name: "zero", DX: []int{1}, DY: []int{0}, W: []float64{0}}},
			{"NaN 权重", &Kernel{Name: "nan", DX: []int{1}, DY: []int{0}, W: []float64{math.NaN()}}},
			{"原点偏移", &Kernel{Name: "origin", DX: []int{0}, DY: []int{0}, W: []float64{1}}},
			{"反向偏移", &Kernel{Name: "back", DX: []int{-1}, DY: []int{0}, W: []float64{1}}},
			{"上探偏移", &Kernel{Name: "up", DX: []int{0}, DY: []int{-1}, W: []float64{1}}},
			{"重复偏移", &Kernel{Name: "dup", DX: []int{1, 1}, DY: []int{0, 0}, W: []float64{0.5, 0.5}}},
			{"权重和超一", &Kernel{Name: "over", DX: []int{1, 2}, DY: []int{0, 0}, W: []float64{0.7, 0.7}}},
		}
		for _, c := range bad {
			if _, err := Dither(g, DitherOptions{Bits: 2, Strength: 1, Kernel: c.k}); !errors.Is(err, ErrKernel) {
				t.Fatalf("%s: 期望 ErrKernel，实际 %v", c.name, err)
			}
		}
	})

	t.Run("强度为负", func(t *testing.T) {
		for _, s := range []float64{-1, -0.001, math.NaN(), math.Inf(1)} {
			if _, err := Dither(g, DitherOptions{Bits: 2, Strength: s, Kernel: good}); !errors.Is(err, ErrStrength) {
				t.Fatalf("强度 %v: 期望 ErrStrength，实际 %v", s, err)
			}
		}
	})

	t.Run("位深越界", func(t *testing.T) {
		for _, b := range []int{-1, 0, 9, 16} {
			if _, err := Dither(g, DitherOptions{Bits: b, Strength: 1, Kernel: good}); !errors.Is(err, ErrBits) {
				t.Fatalf("位深 %d: 期望 ErrBits，实际 %v", b, err)
			}
		}
	})

	t.Run("未知核", func(t *testing.T) {
		if _, err := LookupKernel("no-such-kernel"); !errors.Is(err, ErrKernel) {
			t.Fatalf("期望 ErrKernel，实际 %v", err)
		}
	})
}

func TestDitherSnakeOrder(t *testing.T) {
	// 单抽头 (1,0) 权重 1：误差只沿扫描方向前进。
	// 蛇形下第二行从右向左，输出应与第一行镜像；逐行同向则两行相同。
	k := &Kernel{Name: "carry", DX: []int{1}, DY: []int{0}, W: []float64{1}}
	g := solid(4, 2, 128)
	out, err := Dither(g, DitherOptions{Bits: 1, Strength: 1, Kernel: k})
	if err != nil {
		t.Fatal(err)
	}
	wantRow0 := []uint8{255, 0, 255, 0}
	wantRow1 := []uint8{0, 255, 0, 255}
	for x := 0; x < 4; x++ {
		if out.Pix[x] != wantRow0[x] {
			t.Fatalf("第 0 行第 %d 列 = %d，期望 %d（输出 %v）", x, out.Pix[x], wantRow0[x], out.Pix[:4])
		}
		if out.Pix[4+x] != wantRow1[x] {
			t.Fatalf("第 1 行第 %d 列 = %d，期望 %d（输出 %v）", x, out.Pix[4+x], wantRow1[x], out.Pix[4:])
		}
	}
}

func TestDitherCoverage(t *testing.T) {
	k := fsKernel(t)

	t.Run("灰阶", func(t *testing.T) {
		g := gradient(128, 64)
		out, err := Dither(g, DitherOptions{Bits: 3, Strength: 1, Kernel: k})
		if err != nil {
			t.Fatal(err)
		}
		if n := LevelsUsed(out); n > 8 {
			t.Fatalf("色阶超限 %d", n)
		}
		pal := paletteForLevels(8)
		allowed := make(map[uint8]bool)
		for _, v := range pal {
			allowed[v] = true
		}
		for i, v := range out.Pix {
			if !allowed[v] {
				t.Fatalf("第 %d 像素 %d 不在色阶表内", i, v)
			}
		}
		maeRef, err := MeanAbsError(out, mustQuantize(t, g, 8))
		if err != nil {
			t.Fatal(err)
		}
		if maeRef > 24 {
			t.Fatalf("对参考图平均误差过大 %.2f", maeRef)
		}
		ref := mustQuantize(t, g, 8)
		maePlain, _ := MeanAbsError(ref, g)
		if maeOrig, _ := MeanAbsError(out, g); maeOrig > maePlain*1.5 {
			t.Fatalf("抖动后对原图误差 %.2f 反比直接量化 %.2f 更差", maeOrig, maePlain)
		}
		if d := math.Abs(meanOf(out) - meanOf(g)); d > 1 {
			t.Fatalf("平均亮度漂移 %.2f", d)
		}
	})

	t.Run("纯色", func(t *testing.T) {
		g := solid(64, 64, 128)
		out, err := Dither(g, DitherOptions{Bits: 1, Strength: 1, Kernel: k})
		if err != nil {
			t.Fatal(err)
		}
		zeros, whites := 0, 0
		for _, v := range out.Pix {
			switch v {
			case 0:
				zeros++
			case 255:
				whites++
			default:
				t.Fatalf("非黑白像素 %d", v)
			}
		}
		if zeros == 0 || whites == 0 {
			t.Fatalf("纯色 128 应抖成黑白两色，实际 0=%d 255=%d", zeros, whites)
		}
		if d := math.Abs(meanOf(out) - 128); d > 6 {
			t.Fatalf("纯色平均亮度漂移 %.2f", d)
		}
	})

	t.Run("细密纹理", func(t *testing.T) {
		g := checker(32, 32)
		out, err := Dither(g, DitherOptions{Bits: 1, Strength: 1, Kernel: k})
		if err != nil {
			t.Fatal(err)
		}
		for i := range out.Pix {
			if out.Pix[i] != g.Pix[i] {
				t.Fatalf("黑白棋盘在 1 位下应逐像素保持，第 %d 像素 %d→%d", i, g.Pix[i], out.Pix[i])
			}
		}
		tex := fineTexture(64, 48)
		out, err = Dither(tex, DitherOptions{Bits: 2, Strength: 0.8, Kernel: k})
		if err != nil {
			t.Fatal(err)
		}
		if n := LevelsUsed(out); n > 4 {
			t.Fatalf("纹理色阶超限 %d", n)
		}
	})

	t.Run("单像素", func(t *testing.T) {
		for _, v := range []uint8{0, 1, 127, 128, 254, 255} {
			g := solid(1, 1, v)
			out, err := Dither(g, DitherOptions{Bits: 1, Strength: 1, Kernel: k})
			if err != nil {
				t.Fatalf("值 %d: %v", v, err)
			}
			if len(out.Pix) != 1 {
				t.Fatalf("单像素输出长度 %d", len(out.Pix))
			}
			if out.Pix[0] != 0 && out.Pix[0] != 255 {
				t.Fatalf("单像素输出 %d 不在 1 位色阶", out.Pix[0])
			}
		}
	})

	t.Run("全黑", func(t *testing.T) {
		g := solid(40, 30, 0)
		out, err := Dither(g, DitherOptions{Bits: 1, Strength: 1, Kernel: k})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range out.Pix {
			if v != 0 {
				t.Fatalf("全黑图第 %d 像素变 %d", i, v)
			}
		}
		if bc := CountBands(out); bc.ScanLines != 0 || bc.LongRuns != 0 {
			t.Fatalf("全黑图应无环带，实际 %+v", bc)
		}
	})
}

func TestDitherZeroStrengthNoDiffusion(t *testing.T) {
	g := gradient(64, 4)
	out, err := Dither(g, DitherOptions{Bits: 2, Strength: 0, Kernel: fsKernel(t)})
	if err != nil {
		t.Fatal(err)
	}
	ref := mustQuantize(t, g, 4)
	for i := range out.Pix {
		if out.Pix[i] != ref.Pix[i] {
			t.Fatalf("强度 0 应等于逐像素量化，第 %d 像素 %d≠%d", i, out.Pix[i], ref.Pix[i])
		}
	}
}

func meanOf(g *Gray) float64 {
	sum := 0.0
	for _, v := range g.Pix {
		sum += float64(v)
	}
	return sum / float64(len(g.Pix))
}

func mustQuantize(t *testing.T, g *Gray, levels int) *Gray {
	t.Helper()
	q, err := Quantize(g, levels)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

func TestCountBandsBothMetrics(t *testing.T) {
	// 64×64 左右两半：口径 A 数出 64 条整列；口径 B 的行向游程每行 2 条
	// × 64 行，加上列向 64 条，共 192。
	stripes := mustNew(t, 64, 64)
	fill(stripes, func(x, y int) uint8 {
		if x < 32 {
			return 0
		}
		return 255
	})
	if bc := CountBands(stripes); bc.ScanLines != 64 || bc.LongRuns != 192 {
		t.Fatalf("双色竖条：期望 A=64 B=192，实际 %+v", bc)
	}

	// 1 像素宽棋盘：无整行整列，也没有达到长度阈值的游程。
	if bc := CountBands(checker(64, 64)); bc != (BandCounts{}) {
		t.Fatalf("棋盘：期望 0/0，实际 %+v", bc)
	}
	// 细密纹理：高频噪声，两口径都应接近 0。
	if bc := CountBands(fineTexture(64, 64)); bc != (BandCounts{}) {
		t.Fatalf("细密纹理：期望 0/0，实际 %+v", bc)
	}
	// 纯色、全黑、单像素：单色图不算环带；单像素没有长度达标的游程。
	for name, g := range map[string]*Gray{
		"纯色":  solid(32, 32, 77),
		"全黑":  solid(32, 32, 0),
		"单像素": solid(1, 1, 200),
	} {
		if bc := CountBands(g); bc != (BandCounts{}) {
			t.Fatalf("%s：期望 0/0，实际 %+v", name, bc)
		}
	}
	// 4 段水平条纹（32×32，口径 B 阈值 32）：口径 A 数 32 条整行；
	// 口径 B 行向游程每行 1 条 × 32 行，列向每段只有 8 像素、不到阈值，共 32。
	hstripes := mustNew(t, 32, 32)
	fill(hstripes, func(x, y int) uint8 { return uint8((y / 8) * 85) })
	if bc := CountBands(hstripes); bc.ScanLines != 32 || bc.LongRuns != 32 {
		t.Fatalf("四段横条：期望 A=32 B=32，实际 %+v", bc)
	}
	// 局部平台：128×128 图里放一块 48×48 的同值方块（阈值 32），
	// 不贯穿整行/整列。口径 A 看不见它，口径 B 数出行向 48 + 列向 48 = 96。
	plateau := mustNew(t, 128, 128)
	for i := range plateau.Pix {
		plateau.Pix[i] = uint8(20 + i%7)
	}
	for y := 40; y < 88; y++ {
		for x := 40; x < 88; x++ {
			plateau.Pix[y*128+x] = 200
		}
	}
	if bc := CountBands(plateau); bc.ScanLines != 0 || bc.LongRuns != 96 {
		t.Fatalf("局部平台：期望 A=0 B=96，实际 %+v", bc)
	}
	// 单像素宽的长竖线（64×64，阈值 32）：口径 A 数出 64 条整列；
	// 口径 B 行向每行只有右侧 33 像素达标（左侧 30 不到阈值）× 64 行，
	// 加上列向 64 条，共 128。
	line := solid(64, 64, 0)
	for y := 0; y < 64; y++ {
		line.Pix[y*64+30] = 255
	}
	if bc := CountBands(line); bc.ScanLines != 64 || bc.LongRuns != 128 {
		t.Fatalf("单像素竖线：期望 A=64 B=128，实际 %+v", bc)
	}
}

func TestBandReduction(t *testing.T) {
	if r, ok := BandReduction(100, 40); !ok || math.Abs(r-0.6) > 1e-9 {
		t.Fatalf("降幅计算错误 %v %v", r, ok)
	}
	if _, ok := BandReduction(0, 0); ok {
		t.Fatal("基线为零时应报告无参考基线")
	}
}
