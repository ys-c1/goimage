package img

import (
	"errors"
	"math"
	"strings"
	"testing"
)

const tileW = 32

// ---- 图案生成器：输入全局浮点坐标，输出灰度 ----

func solidPattern(_, _ float64) uint8 { return 100 }

// 平滑斜坡，不做 mod 256 回绕（回绕会产生与错位无关的硬断崖）。
func gradientPattern(x, y float64) uint8 {
	v := 0.7*x + 0.5*y
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return uint8(math.Round(v))
}

func highFreqPattern(x, y float64) uint8 {
	checker := 40.0
	if (int(math.Floor(x/4))+int(math.Floor(y/4)))%2 == 0 {
		checker = -40
	}
	v := 128 + 70*math.Sin(0.9*x)*math.Sin(0.7*y) + checker
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return uint8(math.Round(v))
}

// sinePattern 两个方向都有强梯度，便于错位恢复。
func sinePattern(x, y float64) uint8 {
	v := 128 + 90*math.Sin(0.35*y)*math.Cos(0.31*x) + 20*math.Sin(0.12*x+0.17*y)
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return uint8(math.Round(v))
}

// cutTiles 按规则网格从无限图案切图；shiftY(c)/shiftX(r) 给每块施加错位（像素）。
func cutTiles(pattern func(x, y float64) uint8, cols, rows, W, H int,
	shiftY func(c int) float64, shiftX func(r int) float64) []*Gray {
	tiles := make([]*Gray, cols*rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			g, _ := New(W, H)
			sy := shiftY(c)
			sx := shiftX(r)
			for y := 0; y < H; y++ {
				for x := 0; x < W; x++ {
					gx := float64(c*W+x) - sx
					gy := float64(r*H+y) - sy
					g.Pix[y*W+x] = pattern(gx, gy)
				}
			}
			tiles[r*cols+c] = g
		}
	}
	return tiles
}

func noShift(_ int) float64 { return 0 }

// ---- 五类用例 ----

func TestStitchSolid(t *testing.T) {
	tiles := cutTiles(solidPattern, 2, 2, tileW, tileW, noShift, noShift)
	out, rep, err := Stitch(tiles, 2)
	if err != nil {
		t.Fatal(err)
	}
	if out.W != 64 || out.H != 64 {
		t.Fatalf("成品尺寸错误 %dx%d", out.W, out.H)
	}
	for _, s := range rep.Seams {
		if !s.Flat || s.Delta != 0 {
			t.Fatalf("同色接缝应平坦且偏移为零: %+v", s)
		}
	}
	assertInvariant(t, rep)
}

func TestStitchGradient(t *testing.T) {
	for _, tc := range []struct {
		name       string
		cols, rows int
	}{{"四档2x2", 2, 2}, {"十六档4x4", 4, 4}} {
		t.Run(tc.name, func(t *testing.T) {
			tiles := cutTiles(gradientPattern, tc.cols, tc.rows, tileW, tileW, noShift, noShift)
			out, rep, err := Stitch(tiles, tc.cols)
			if err != nil {
				t.Fatal(err)
			}
			if out.W != tc.cols*tileW || out.H != tc.rows*tileW {
				t.Fatalf("成品尺寸错误")
			}
			for _, s := range rep.Seams {
				if math.Abs(s.Delta) > maxSeamShift {
					t.Fatalf("接缝偏移越界: %+v", s)
				}
			}
			assertInvariant(t, rep)
		})
	}
}

func TestStitchHighFreq(t *testing.T) {
	tiles := cutTiles(highFreqPattern, 2, 2, tileW, tileW, noShift, noShift)
	_, rep, err := Stitch(tiles, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertInvariant(t, rep)
	if rep.ESeam1 > rep.ESeam0+1e-6 {
		t.Fatalf("接缝能量上升: 前 %.4f 后 %.4f", rep.ESeam0, rep.ESeam1)
	}
}

func TestStitchMisaligned(t *testing.T) {
	const s = 0.35 // 块间真实错位 0.35px，落在 ±0.5px 内
	tiles := cutTiles(sinePattern, 2, 2, tileW, tileW,
		func(c int) float64 { return s * float64(c) }, // 列块沿 y 错位
		func(r int) float64 { return s * float64(r) }) // 行块沿 x 错位
	_, rep, err := Stitch(tiles, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, seam := range rep.Seams {
		// 采样约定 b(s-δ)：右块沿缝向下/向右滑 s，解出 δ=-s。
		// 2x2 时每条缝两侧只有 1 个块对，端部有夹取噪声，容差 0.1px。
		if math.Abs(seam.Delta+s) > 0.1 {
			t.Fatalf("%s%d 恢复偏移 %.4f 与真值 -%.4f 偏差过大",
				seam.Orientation, seam.Index, seam.Delta, s)
		}
		if math.Abs(seam.Delta) > maxSeamShift {
			t.Fatalf("接缝偏移越界: %+v", seam)
		}
	}
	assertInvariant(t, rep)
}

func TestStitchSingleTilePassthrough(t *testing.T) {
	src, _ := New(6, 4)
	for i := range src.Pix {
		src.Pix[i] = uint8(i*7 + 3)
	}
	out, rep, err := Stitch([]*Gray{src}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if out.W != src.W || out.H != src.H {
		t.Fatal("单块成品尺寸变化")
	}
	for i := range src.Pix {
		if out.Pix[i] != src.Pix[i] {
			t.Fatalf("单块路径像素被改动: 位置 %d 原 %d 出 %d", i, src.Pix[i], out.Pix[i])
		}
	}
	if len(rep.Seams) != 0 {
		t.Fatal("单块不应有接缝")
	}
	if math.Abs(rep.EAfter-rep.EBefore) > 1e-12 {
		t.Fatalf("单块能量应完全相等: 前 %.6f 后 %.6f", rep.EBefore, rep.EAfter)
	}
}

// ---- 三类错误，均须带块号 ----

func TestStitchErrors(t *testing.T) {
	t.Run("空块报块号", func(t *testing.T) {
		good, _ := New(8, 8)
		bad := &Gray{W: 8, H: 8} // Pix 为空
		_, _, err := Stitch([]*Gray{good, bad}, 2)
		if !errors.Is(err, ErrTileBlank) || !strings.Contains(err.Error(), "块号 1") {
			t.Fatalf("空块错误信息不合格: %v", err)
		}
	})

	t.Run("尺寸不一致报块号", func(t *testing.T) {
		a, _ := New(8, 8)
		b, _ := New(8, 6)
		_, _, err := Stitch([]*Gray{a, b}, 2)
		if !errors.Is(err, ErrTileSize) || !strings.Contains(err.Error(), "块号 1") {
			t.Fatalf("尺寸错误信息不合格: %v", err)
		}
	})

	t.Run("空列表", func(t *testing.T) {
		if _, _, err := Stitch(nil, 1); !errors.Is(err, ErrNoTiles) {
			t.Fatalf("空列表错误不合格: %v", err)
		}
	})

	t.Run("偏移越界按失败处理", func(t *testing.T) {
		const big = 0.9 // 超出 ±0.5px
		tiles := cutTiles(sinePattern, 2, 2, tileW, tileW,
			func(c int) float64 { return big * float64(c) },
			func(r int) float64 { return big * float64(r) })
		_, _, err := Stitch(tiles, 2)
		if !errors.Is(err, ErrOffsetRange) {
			t.Fatalf("越界应失败: %v", err)
		}
	})
}

// assertInvariant 不变量：拼接后整体梯度能量不得高于原块之和。
func assertInvariant(t *testing.T, rep *StitchReport) {
	t.Helper()
	tol := rep.EBefore * 1e-9
	if rep.EAfter > rep.EBefore+tol {
		t.Fatalf("能量不变量被破坏: 原块之和 %.6f > 成品 %.6f（差 %.6f）",
			rep.EBefore, rep.EAfter, rep.EAfter-rep.EBefore)
	}
}

// ---- 两档图块数的耗时基准（稳定计时，排除一次性噪声） ----

func BenchmarkStitch(b *testing.B) {
	for _, tier := range []struct {
		label      string
		cols, rows int
	}{
		{"4块2x2", 2, 2},
		{"16块4x4", 4, 4},
	} {
		tiles := cutTiles(sinePattern, tier.cols, tier.rows, tileW, tileW,
			func(c int) float64 { return 0.35 * float64(c) },
			func(r int) float64 { return 0.35 * float64(r) })
		b.Run(tier.label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, _, err := Stitch(tiles, tier.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestStitchReport(t *testing.T) {
	cases := []struct {
		name    string
		pattern func(x, y float64) uint8
		s       float64 // 错位幅度，0 表示不错位
	}{
		{"同色", solidPattern, 0},
		{"渐变", gradientPattern, 0},
		{"高频纹理", highFreqPattern, 0},
		{"错位0.35px", sinePattern, 0.35},
	}
	tiers := []struct {
		label      string
		cols, rows int
	}{
		{"四档 2x2", 2, 2},
		{"十六档 4x4", 4, 4},
	}
	for _, tier := range tiers {
		for _, tc := range cases {
			tiles := cutTiles(tc.pattern, tier.cols, tier.rows, tileW, tileW,
				func(c int) float64 { return tc.s * float64(c) },
				func(r int) float64 { return tc.s * float64(r) })
			_, rep, err := Stitch(tiles, tier.cols)
			if err != nil {
				t.Fatalf("%s/%s 拼接失败: %v", tier.label, tc.name, err)
			}
			t.Logf("")
			t.Logf("==== %s | 用例:%s | 块尺寸 %dx%d | 耗时 %v ====",
				tier.label, tc.name, rep.TileW, rep.TileH, rep.Duration)
			t.Logf("接缝偏移表（共 %d 条，允许范围 ±%.1fpx）:", len(rep.Seams), maxSeamShift)
			for _, sm := range rep.Seams {
				t.Logf("  %s%d  块 %d|%d  偏移 %+.4fpx  平坦=%v",
					sm.Orientation, sm.Index, sm.LeftBlock, sm.RightBlock, sm.Delta, sm.Flat)
			}
			t.Logf("能量  原块内部 %.4f | 原块接缝错配 %.4f | 合计 E0=%.4f",
				rep.EInternal0, rep.ESeam0, rep.EBefore)
			t.Logf("能量  成品内部 %.4f | 成品接缝     %.4f | 合计 E1=%.4f",
				rep.EInternal1, rep.ESeam1, rep.EAfter)
			t.Logf("能量差 ΔE = E1-E0 = %+.6f（不变量要求 ΔE ≤ 0）", rep.EAfter-rep.EBefore)
		}
	}
}
