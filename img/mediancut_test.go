package img

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ---- 测试素材构造：渐变、照片、图标、纯色、单像素 -------------------------

func makeGradient16(w, h int) *Gray16 {
	// 浮点插值的非对齐渐变，步长不落在八位阶边界上，更贴近真实素材。
	g, _ := NewGray16(w, h)
	for i := range g.Pix {
		x := i % w
		g.Pix[i] = uint16(math.Round(float64(x) * 65535.0 / float64(w-1)))
	}
	return g
}

func makePhoto16(w, h int) *Gray16 {
	// 合成“照片”：多频率正弦叠加 + 平滑明暗场，模拟连续色调。
	g, _ := NewGray16(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			fx, fy := float64(x)/float64(w), float64(y)/float64(h)
			v := 0.5 + 0.25*math.Sin(fx*18.0) + 0.15*math.Cos(fy*23.0) +
				0.10*math.Sin((fx+fy)*31.0)
			if v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			g.Pix[y*w+x] = uint16(v * 65535)
		}
	}
	return g
}

func makeIcon16(w, h int) *Gray16 {
	// 图标：少量离散色 + 锐利边缘（圆环 + 十字）。
	g, _ := NewGray16(w, h)
	cx, cy, r := w/2, h/2, w/3
	tones := []uint16{0, 12000, 34000, 55000, 65535}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx, dy := x-cx, y-cy
			d := dx*dx + dy*dy
			switch {
			case d < r*r/4:
				g.Pix[y*w+x] = tones[4]
			case d < r*r:
				g.Pix[y*w+x] = tones[2]
			case x == cx || y == cy:
				g.Pix[y*w+x] = tones[3]
			default:
				g.Pix[y*w+x] = tones[1]
			}
		}
	}
	return g
}

func makeSolid16(w, h int, tone uint16) *Gray16 {
	g, _ := NewGray16(w, h)
	for i := range g.Pix {
		g.Pix[i] = tone
	}
	return g
}

func makeSinglePixel16() *Gray16 {
	g, _ := NewGray16(1, 1)
	g.Pix[0] = 40000
	return g
}

func testImages() map[string]*Gray16 {
	return map[string]*Gray16{
		"渐变":  makeGradient16(128, 96),
		"照片":  makePhoto16(128, 96),
		"图标":  makeIcon16(64, 64),
		"纯色":  makeSolid16(64, 64, 32768),
		"单像素": makeSinglePixel16(),
	}
}

// ---- 主实现 vs 手算参考：逐像素对照 ---------------------------------------

func TestMedianCutMatchesHandReference(t *testing.T) {
	for name, g := range testImages() {
		for _, nb := range []int{1, 2, 7, 16, 64, 256} {
			out, stats, err := MedianCut16(g, nb)
			if err != nil {
				t.Fatalf("%s nb=%d 意外错误: %v", name, nb, err)
			}
			refBuckets, refPal := refMedianCut(append([]uint16(nil), g.Pix...), nb)
			if stats.Buckets != refBuckets {
				t.Fatalf("%s nb=%d 原始桶数不一致: 生产=%d 手算=%d",
					name, nb, stats.Buckets, refBuckets)
			}
			if len(stats.Palette) != len(refPal) {
				t.Fatalf("%s nb=%d 去重后调色板项数不一致: 生产=%d 手算=%d",
					name, nb, len(stats.Palette), len(refPal))
			}
			for i, p := range stats.Palette {
				if p != refPal[i] {
					t.Fatalf("%s nb=%d 第 %d 项调色板不一致: 生产=%d 手算=%d",
						name, nb, i, p, refPal[i])
				}
			}
			// 逐像素：生产实现的最近邻必须与暴力手算映射一致。
			for i, v := range g.Pix {
				want := refMapBruteForce(v, refPal)
				if out.Pix[i] != want {
					t.Fatalf("%s nb=%d 像素 %d 映射不一致: 生产=%d 手算=%d",
						name, nb, i, out.Pix[i], want)
				}
			}
		}
	}
}

// 手算数值样例：四个像素、两桶，可在纸面复算。
func TestHandComputedTinyCase(t *testing.T) {
	g, _ := NewGray16(4, 1)
	g.Pix = []uint16{0, 10280, 51400, 65535} // 对应八位 0、40、200、255
	_, pal := refMedianCut(append([]uint16(nil), g.Pix...), 2)
	// 排序后下中位 pivot=10280，切点在 {0,10280}|{51400,65535}。
	// 左桶均值 5140 -> round=20；右桶均值 58467.5 -> round=228。
	want := []uint8{20, 228}
	if len(pal) != 2 || pal[0] != want[0] || pal[1] != want[1] {
		t.Fatalf("手算样例失败，得到 %v，想要 %v", pal, want)
	}
	out, _, err := MedianCut16(g, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range g.Pix {
		if out.Pix[i] != refMapBruteForce(v, pal) {
			t.Fatalf("像素 %d 与手算映射不符", i)
		}
	}
}

// ---- 色差上界：实测最大色差不得超过声明上界 -------------------------------

func TestErrorBoundHolds(t *testing.T) {
	for name, g := range testImages() {
		for _, nb := range []int{2, 16, 256} {
			_, stats, err := MedianCut16(g, nb)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if stats.MaxErr16 > stats.Bound16 {
				t.Fatalf("%s nb=%d 实测色差 %d 超过上界 %d",
					name, nb, stats.MaxErr16, stats.Bound16)
			}
			if stats.Bound16 < 0 || stats.MaxErr16 < 0 {
				t.Fatalf("%s 色差统计为负", name)
			}
			// 独立复算一遍最大色差，不信任统计路径自身。
			maxD := 0
			for _, v := range g.Pix {
				want := refMapBruteForce(v, stats.Palette)
				dd := absInt(int(v) - int(p8to16(want)))
				if dd > maxD {
					maxD = dd
				}
			}
			if maxD != stats.MaxErr16 {
				t.Fatalf("%s nb=%d 最大色差复算不符: %d vs %d",
					name, nb, maxD, stats.MaxErr16)
			}
		}
	}
}

// ---- 桶数约束与去重上限 ----------------------------------------------------

func TestBucketCapAndClamp(t *testing.T) {
	solid := makeSolid16(16, 16, 12345)
	out, stats, err := MedianCut16(solid, 256)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Buckets != 1 || len(stats.Palette) != 1 {
		t.Fatalf("纯色应只产生一个桶，得到 %d/%d", stats.Buckets, len(stats.Palette))
	}
	// 纯色输出必须全部相同。
	for i := 1; i < len(out.Pix); i++ {
		if out.Pix[i] != out.Pix[0] {
			t.Fatal("纯色量化输出不统一")
		}
	}
	// 桶数上限生效：渐变在 256 桶下也不能超出八位容量。
	g := makeGradient16(256, 1)
	_, stats, err = MedianCut16(g, 256)
	if err != nil {
		t.Fatal(err)
	}
	if len(stats.Palette) > 256 {
		t.Fatalf("调色板超出八位容量: %d", len(stats.Palette))
	}
}

// ---- 错误码：桶数为零、色阶非法、图为空 ------------------------------------

func TestErrorCodes(t *testing.T) {
	g := makeGradient16(8, 8)

	if _, _, err := MedianCut16(g, 0); !errors.Is(err, ErrBuckets) {
		t.Fatalf("桶数为零应返回 ErrBuckets，得到 %v", err)
	}
	if _, _, err := MedianCut16(g, -1); !errors.Is(err, ErrBuckets) {
		t.Fatalf("桶数为负应返回 ErrBuckets，得到 %v", err)
	}
	if _, _, err := MedianCut16(g, 257); !errors.Is(err, ErrBuckets) {
		t.Fatalf("桶数超 256 应返回 ErrBuckets，得到 %v", err)
	}
	// 既有八位路径的色阶非法错误保持独立。
	if _, err := Quantize(&Gray{W: 2, H: 2, Pix: make([]uint8, 4)}, 0); !errors.Is(err, ErrLevels) {
		t.Fatalf("色阶非法应返回 ErrLevels，得到 %v", err)
	}
	if _, err := Quantize(&Gray{W: 2, H: 2, Pix: make([]uint8, 4)}, 300); !errors.Is(err, ErrLevels) {
		t.Fatalf("色阶超界应返回 ErrLevels，得到 %v", err)
	}
	// 空图：nil 与零尺寸都要报 ErrEmpty。
	if _, _, err := MedianCut16(nil, 16); !errors.Is(err, ErrEmpty) {
		t.Fatalf("nil 图应返回 ErrEmpty，得到 %v", err)
	}
	empty := &Gray16{}
	if _, _, err := MedianCut16(empty, 16); !errors.Is(err, ErrEmpty) {
		t.Fatalf("空图应返回 ErrEmpty，得到 %v", err)
	}
	if err := (&Gray16{}).SavePGM16(filepath.Join(t.TempDir(), "x.pgm")); !errors.Is(err, ErrEmpty) {
		t.Fatalf("空图导出应返回 ErrEmpty，得到 %v", err)
	}
}

// ---- 八位直通路径行为不变：固定黄金值 --------------------------------------

func TestEightBitPassthroughUnchanged(t *testing.T) {
	g, _ := New(4, 1)
	g.Pix = []uint8{0, 60, 130, 250}
	out, err := Quantize(g, 4)
	if err != nil {
		t.Fatal(err)
	}
	// step = 255/3 = 85；量化结果锁定为当前发布行为。
	want := []uint8{0, 0, 85, 170}
	for i := range want {
		if out.Pix[i] != want[i] {
			t.Fatalf("八位直通路径被改动: 位置 %d 得到 %d 想要 %d", i, out.Pix[i], want[i])
		}
	}
}

// ---- 十六位 PGM 往返 -------------------------------------------------------

func TestGray16PGMRoundTrip(t *testing.T) {
	g := makePhoto16(32, 16)
	p := filepath.Join(t.TempDir(), "photo16.pgm")
	if err := g.SavePGM16(p); err != nil {
		t.Fatal(err)
	}
	got, err := LoadPGM16(p)
	if err != nil {
		t.Fatal(err)
	}
	if got.W != g.W || got.H != g.H || len(got.Pix) != len(g.Pix) {
		t.Fatal("尺寸不符")
	}
	for i := range g.Pix {
		if got.Pix[i] != g.Pix[i] {
			t.Fatalf("像素 %d 不符: %d vs %d", i, got.Pix[i], g.Pix[i])
		}
	}
	// 八位 PGM 不能被十六位加载器接受。
	p8 := filepath.Join(t.TempDir(), "p8.pgm")
	if err := (&Gray{W: 2, H: 2, Pix: []uint8{1, 2, 3, 4}}).SavePGM(p8); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPGM16(p8); !errors.Is(err, ErrHeader) {
		t.Fatalf("八位 PGM 应被十六位加载器拒绝，得到 %v", err)
	}
	// 坏头也要拒绝。
	bad := filepath.Join(t.TempDir(), "bad.pgm")
	if err := os.WriteFile(bad, []byte("P9\n2 2\n65535\nxxxx"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPGM16(bad); !errors.Is(err, ErrHeader) {
		t.Fatalf("坏头应返回 ErrHeader，得到 %v", err)
	}
}

// ---- 截断 vs 中位切分：低桶数下截断色带更重的事实性检查 --------------------

// makeNonUniformRamp16 构造强偏态分布：前 90% 像素挤在 0..8000 的窄密集区
// （相邻仅 40 阶），后 10% 散在 8000..65535 的稀疏区。等间距截断把桶均匀
// 撒在全范围，密集区分不到几个桶；中位切分则把多数桶自适应投到密集区。
func makeNonUniformRamp16(w int) *Gray16 {
	g, _ := NewGray16(w, 1)
	dense := w * 9 / 10
	v := 0
	for x := 0; x < w; x++ {
		g.Pix[x] = uint16(v)
		if x < dense {
			v += 40
			if v > 8000 {
				v = 8000
			}
		} else {
			v += 5000
			if v > 65535 {
				v = 65535
			}
		}
	}
	return g
}

// countRuns 统计输出中连续同值平台的数量。
func countRuns(p []uint8) int {
	runs := 1
	for i := 1; i < len(p); i++ {
		if p[i] != p[i-1] {
			runs++
		}
	}
	return runs
}

func meanAbsErr16(g *Gray16, q *Gray) float64 {
	sum := 0.0
	for i, v := range g.Pix {
		sum += float64(absInt(int(v) - int(p8to16(q.Pix[i]))))
	}
	return sum / float64(len(g.Pix))
}

func TestMedianCutBeatsTruncationOnGradient(t *testing.T) {
	const k = 16
	g := makeNonUniformRamp16(200)
	denseCount := len(g.Pix) * 9 / 10

	out, mc, err := MedianCut16(g, k)
	if err != nil {
		t.Fatal(err)
	}
	uni := UniformTruncate16(g, k)

	// 同桶数预算下，密集区（前 90% 像素）的平均与最大色差，
	// 自适应切分都必须优于等间距截断。
	var sumMC, sumUni float64
	maxMC, maxUni := 0, 0
	for i := 0; i < denseCount; i++ {
		dMC := absInt(int(g.Pix[i]) - int(p8to16(out.Pix[i])))
		dUni := absInt(int(g.Pix[i]) - int(p8to16(uni.Pix[i])))
		sumMC += float64(dMC)
		sumUni += float64(dUni)
		if dMC > maxMC {
			maxMC = dMC
		}
		if dUni > maxUni {
			maxUni = dUni
		}
	}
	meanMC, meanUni := sumMC/float64(denseCount), sumUni/float64(denseCount)
	if meanMC >= meanUni {
		t.Fatalf("密集区平均色差未改善: 中位切分=%.1f 等间距=%.1f", meanMC, meanUni)
	}
	if maxMC >= maxUni {
		t.Fatalf("密集区最大色差未改善: 中位切分=%d 等间距=%d", maxMC, maxUni)
	}
	// 全局平均也应改善，因为 90% 的像素都在密集区。
	if mc.MeanAbs16 >= meanAbsErr16(g, uni) {
		t.Fatalf("全局平均色差未改善: 中位切分=%.1f 等间距=%.1f",
			mc.MeanAbs16, meanAbsErr16(g, uni))
	}
	// 切分恰好 k 个平台；等间距截断在密集区只覆盖到极少数阶，平台更长（色带更重）。
	if runs := countRuns(out.Pix); runs != k {
		t.Fatalf("中位切分应产生 %d 个平台，得到 %d", k, runs)
	}
	if countRuns(uni.Pix) >= countRuns(out.Pix) {
		t.Fatalf("等间距截断平台数应少于中位切分（密集区色带更重）")
	}
}

// ---- 公开辅助 API：八位换算、截断、等间距边界 ------------------------------

func TestStats8ConversionAndTruncation(t *testing.T) {
	g, _ := NewGray16(3, 1)
	g.Pix = []uint16{0, 257 * 100, 65535}

	// 直通截断：高八位。
	tr := Truncate16To8(g)
	if tr.W != 3 || tr.H != 1 || len(tr.Pix) != 3 {
		t.Fatalf("截断输出尺寸错误: %+v", tr)
	}
	wantTr := []uint8{0, 100, 255}
	for i := range wantTr {
		if tr.Pix[i] != wantTr[i] {
			t.Fatalf("截断位置 %d 得到 %d 想要 %d", i, tr.Pix[i], wantTr[i])
		}
	}

	_, st, err := MedianCut16(g, 3)
	if err != nil {
		t.Fatal(err)
	}
	// 八位换算方法与十六位统计严格对应（除以 257）。
	if math.Abs(st.MaxErr8()-float64(st.MaxErr16)/257.0) > 1e-9 {
		t.Fatal("MaxErr8 换算不符")
	}
	if math.Abs(st.Bound8()-float64(st.Bound16)/257.0) > 1e-9 {
		t.Fatal("Bound8 换算不符")
	}
	if math.Abs(st.MeanAbs8()-st.MeanAbs16/257.0) > 1e-9 {
		t.Fatal("MeanAbs8 换算不符")
	}
	if math.Abs(st.RMS8()-st.RMS16/257.0) > 1e-9 {
		t.Fatal("RMS8 换算不符")
	}
}

func TestUniformTruncate(t *testing.T) {
	g, _ := NewGray16(4, 1)
	g.Pix = []uint16{0, 16384, 32768, 65535}

	// k=1 退化为单一中灰输出。
	u1 := UniformTruncate16(g, 1)
	for _, p := range u1.Pix {
		if p != 128 {
			t.Fatalf("单桶等间距应输出 128，得到 %d", p)
		}
	}
	// k=2 应只有 0 与 255 两个输出阶。
	u2 := UniformTruncate16(g, 2)
	seen := map[uint8]bool{}
	for _, p := range u2.Pix {
		seen[p] = true
		if p != 0 && p != 255 {
			t.Fatalf("两桶等间距只能输出 0/255，出现 %d", p)
		}
	}
	if len(seen) != 2 {
		t.Fatalf("两桶等间距应产生两个阶，得到 %d", len(seen))
	}
}

func TestRound16to8Clamp(t *testing.T) {
	if round16to8(-100) != 0 {
		t.Fatal("负值应夹到 0")
	}
	if round16to8(70000) != 255 {
		t.Fatal("超界应夹到 255")
	}
	if round16to8(257*128) != 128 {
		t.Fatal("正常取整错误")
	}
}

func TestNewGray16RejectsZero(t *testing.T) {
	if _, err := NewGray16(0, 2); !errors.Is(err, ErrSize) {
		t.Fatalf("零宽应返回 ErrSize，得到 %v", err)
	}
}

func TestMalformedPGM16Headers(t *testing.T) {
	dir := t.TempDir()
	cases := map[string][]byte{
		"魔数错误":     []byte("P9\n2 2\n65535\nxxxx"),
		"缺尺寸":      []byte("P5\n2\n65535\n"),
		"maxval为零": []byte("P5\n2 2\n0\nxx"),
		"maxval超界": []byte("P5\n2 2\n70000\nxxxx"),
		"尺寸含字母":    []byte("P5\n2 a\n65535\nxxxx"),
		"数据截断":     []byte("P5\n2 2\n65535\n\x00\x01"),
		"八位maxval": []byte("P5\n2 2\n255\n\x01\x02\x03\x04"),
		"带注释合法":    append([]byte("P5\n# comment line\n2 2\n65535\n"), make([]byte, 8)...),
	}
	for name, data := range cases {
		p := filepath.Join(dir, "h.pgm")
		if err := os.WriteFile(p, data, 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := LoadPGM16(p)
		if name == "带注释合法" {
			if err != nil {
				t.Fatalf("%s 应解析成功，得到 %v", name, err)
			}
			continue
		}
		if !errors.Is(err, ErrHeader) {
			t.Fatalf("%s 应返回 ErrHeader，得到 %v", name, err)
		}
	}
}

// ---- 两档图幅 benchmark：报桶数与耗时 --------------------------------------

func benchmarkSize(b *testing.B, w, h int) {
	g := makePhoto16(w, h)
	b.Run("16桶", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, _, err := MedianCut16(g, 16); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("256桶", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, _, err := MedianCut16(g, 256); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkMedianCut_256x256(b *testing.B) { benchmarkSize(b, 256, 256) }
func BenchmarkMedianCut_512x512(b *testing.B) { benchmarkSize(b, 512, 512) }
