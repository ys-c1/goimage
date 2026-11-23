package img

import (
	"errors"
	"math"
	"testing"
)

// 每条用例源色均以 Display P3 编码值给出。
var gamutCases = []struct {
	name string
	in   RGB
}{
	{"饱和红", RGB{1, 0, 0}},
	{"饱和绿", RGB{0, 1, 0}},
	{"饱和蓝", RGB{0, 0, 1}},
	{"肤色", RGB{0.85, 0.55, 0.4}},
	{"浅肤色", RGB{0.9, 0.6, 0.45}},
	{"中灰", RGB{0.5, 0.5, 0.5}},
	{"暗灰", RGB{0.2, 0.2, 0.2}},
	{"纯黑", RGB{0, 0, 0}},
}

func mustMap(t *testing.T, src RGB, cfg GamutConfig) RGB {
	t.Helper()
	out, err := MapPixel(src, cfg)
	if err != nil {
		t.Fatalf("映射 %v: %v", src, err)
	}
	return out
}

func TestGamutNoHueShift(t *testing.T) {
	cfg := DefaultGamutConfig()
	m, err := newGamutMapper(cfg)
	if err != nil {
		t.Fatal(err)
	}
	const maxHueShift = 0.05 // 度，Oklab 色相
	for _, tc := range gamutCases {
		out := mustMap(t, tc.in, cfg)
		srcLab := m.labOfSource(tc.in)
		outLab := m.labOfTarget(out)
		shift := hueShiftDeg(srcLab, outLab)
		if math.Abs(shift) > maxHueShift {
			t.Errorf("%s: 色相偏移 %.4f° 超过 %.2f°", tc.name, shift, maxHueShift)
		}
		t.Logf("%s: 色相偏移 %.6f°，ΔE_OK %.5f", tc.name, shift, deltaEOK(srcLab, outLab))
	}
}

func TestGamutNoHueShiftUniformRegion(t *testing.T) {
	cfg := DefaultGamutConfig()
	const w, h = 8, 6
	im, err := NewColorImage(w, h)
	if err != nil {
		t.Fatal(err)
	}
	base := RGB{0.9, 0.25, 0.1} // 超域，且接近饱和红
	for i := range im.Pix {
		im.Pix[i] = base
	}
	mapped, err := MapToGamut(im, cfg)
	if err != nil {
		t.Fatal(err)
	}
	first := mapped.Pix[0]
	for i, c := range mapped.Pix {
		if math.Abs(c.R-first.R) > 1e-12 || math.Abs(c.G-first.G) > 1e-12 || math.Abs(c.B-first.B) > 1e-12 {
			t.Fatalf("同色区域内像素 %d 与首像素不一致: %v vs %v", i, c, first)
		}
		if !(c.R >= 0 && c.R <= 1 && c.G >= 0 && c.G <= 1 && c.B >= 0 && c.B <= 1) {
			t.Fatalf("像素 %d 超出编码范围: %v", i, c)
		}
	}
}

// buildGamutGrid 生成覆盖 P3 编码立方体的彩色图，作为超域计数样本。
func buildGamutGrid(t *testing.T) *ColorImage {
	t.Helper()
	steps := []float64{0, 0.08, 0.16, 0.3, 0.5, 0.7, 0.84, 0.92, 1}
	n := len(steps)
	im, err := NewColorImage(n*n*n, 1)
	if err != nil {
		t.Fatal(err)
	}
	idx := 0
	for _, r := range steps {
		for _, g := range steps {
			for _, b := range steps {
				im.Pix[idx] = RGB{r, g, b}
				idx++
			}
		}
	}
	return im
}

func TestGamutOutOfGamutPixelReduction(t *testing.T) {
	cfg := DefaultGamutConfig()
	src := buildGamutGrid(t)
	_, q, err := MeasureMap(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if q.BeforeOutOfGamut == 0 {
		t.Fatal("样本中应存在超域像素")
	}
	t.Logf("超域像素: 映射前 %d/%d (%.1f%%)，映射后编码前残差 %d/%d (%.1f%%)",
		q.BeforeOutOfGamut, q.Total, 100*float64(q.BeforeOutOfGamut)/float64(q.Total),
		q.ResidualOutOfGamut, q.Total, 100*float64(q.ResidualOutOfGamut)/float64(q.Total))
	if q.ResidualOutOfGamut != 0 {
		t.Fatalf("映射后仍有 %d 个像素在编码前越界", q.ResidualOutOfGamut)
	}
	// 给定比例：映射后的超域像素比例不得超过 1%。
	const allowedRatio = 0.01
	if ratio := float64(q.ResidualOutOfGamut) / float64(q.Total); ratio > allowedRatio {
		t.Fatalf("映射后超域比例 %.4f 超过给定比例 %.4f", ratio, allowedRatio)
	}
}

// 域内颜色的“直通”指线性光不经限幅、色相与色差为零；
// 编码值本身会随目标色域的原色而变化，不能要求逐位相同。
func TestGamutInGamutColorsPassThrough(t *testing.T) {
	cfg := DefaultGamutConfig()
	m, err := newGamutMapper(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range gamutCases {
		if m.outOfGamut(tc.in) {
			continue
		}
		out, residual := m.mapEncodedFull(tc.in)
		if residual {
			t.Errorf("%s: 域内颜色被限幅", tc.name)
		}
		if de := deltaEOK(m.labOfSource(tc.in), m.labOfTarget(out)); de > 1e-9 {
			t.Errorf("%s: 域内颜色色差 %.3e", tc.name, de)
		}
	}
	// 纯黑必须逐位保持。
	if black := mustMap(t, RGB{0, 0, 0}, cfg); black != (RGB{0, 0, 0}) {
		t.Fatalf("纯黑未保持: %v", black)
	}
}

func TestSoftClipCurveProperties(t *testing.T) {
	const Cmax, knee = 0.25, 0.9
	c0 := knee * Cmax
	a := Cmax - c0
	if got := softClipChroma(c0, Cmax, knee); math.Abs(got-c0) > 1e-12 {
		t.Fatalf("拐点处不连续: %.15f vs %.15f", got, c0)
	}
	// 左右导数在拐点处应一致（C1 连续）。
	const eps = 1e-9
	left := (softClipChroma(c0, Cmax, knee) - softClipChroma(c0-eps, Cmax, knee)) / eps
	right := (softClipChroma(c0+eps, Cmax, knee) - softClipChroma(c0, Cmax, knee)) / eps
	if math.Abs(left-right) > 1e-4 {
		t.Fatalf("拐点导数不连续: 左 %.6f 右 %.6f", left, right)
	}
	prev := 0.0
	for c := 0.0; c < 2; c += 0.001 {
		v := softClipChroma(c, Cmax, knee)
		if v < prev-1e-15 {
			t.Fatalf("曲线不单调: C=%.4f", c)
		}
		if v > Cmax+1e-12 {
			t.Fatalf("曲线越界: C=%.4f -> %.12f", c, v)
		}
		prev = v
	}
	if softClipChroma(1e6, Cmax, knee) > Cmax+1e-9 {
		t.Fatal("远端未收敛到边界")
	}
	if a <= 0 {
		t.Fatal("软段区间应为正")
	}
}

// TestSoftClipBeatsHardClip 用数值说明为什么不用直接截断：
// 两个相近的超域红色在硬截断后塌成同一颜色，软压缩仍保持区分；
// 硬截断还会造成更大的色相偏移。
func TestSoftClipBeatsHardClip(t *testing.T) {
	cfg := DefaultGamutConfig()
	m, err := newGamutMapper(cfg)
	if err != nil {
		t.Fatal(err)
	}
	a := RGB{1, 0, 0}
	b := RGB{1, 0.12, 0}
	outA := mustMap(t, a, cfg)
	outB := mustMap(t, b, cfg)
	if math.Abs(outA.R-outB.R) < 1e-6 && math.Abs(outA.G-outB.G) < 1e-6 && math.Abs(outA.B-outB.B) < 1e-6 {
		t.Fatal("软压缩把两个超域红色压成同一颜色")
	}

	hard := func(src RGB) RGB {
		lin := m.linearTargetOf(src)
		for i := range lin {
			lin[i] = min(max(lin[i], 0), 1)
		}
		return encodeRGB(lin, cfg.Gamma)
	}
	hardA, hardB := hard(a), hard(b)
	if diffRGB(hardA, hardB) > 1e-12 {
		t.Fatalf("样本构造有误，硬截断后本应相同: %v %v", hardA, hardB)
	}
	srcLabA := m.labOfSource(a)
	srcLabB := m.labOfSource(b)
	softShiftA := math.Abs(hueShiftDeg(srcLabA, m.labOfTarget(outA)))
	softShiftB := math.Abs(hueShiftDeg(srcLabB, m.labOfTarget(outB)))
	hardShiftA := math.Abs(hueShiftDeg(srcLabA, m.labOfTarget(hardA)))
	hardShiftB := math.Abs(hueShiftDeg(srcLabB, m.labOfTarget(hardB)))
	if softShiftA > hardShiftA || softShiftB > hardShiftB {
		t.Fatalf("软压缩色相偏移未优于硬截断: soft %.4f/%.4f hard %.4f/%.4f",
			softShiftA, softShiftB, hardShiftA, hardShiftB)
	}
	t.Logf("硬截断色相偏移 %.4f°/%.4f°，软压缩 %.4f°/%.4f°", hardShiftA, hardShiftB, softShiftA, softShiftB)
}

func TestGamutDoesNotMutateInput(t *testing.T) {
	cfg := DefaultGamutConfig()
	src := buildGamutGrid(t)
	want := make([]RGB, len(src.Pix))
	copy(want, src.Pix)
	if _, err := MapToGamut(src, cfg); err != nil {
		t.Fatal(err)
	}
	for i := range src.Pix {
		if src.Pix[i] != want[i] {
			t.Fatalf("输入图像在像素 %d 被修改: %v -> %v", i, want[i], src.Pix[i])
		}
	}
}

func TestDefaultKneeApplied(t *testing.T) {
	cfg := DefaultGamutConfig()
	explicit := cfg
	explicit.Knee = 0.9
	zero := cfg
	zero.Knee = 0
	for _, src := range []RGB{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} {
		a := mustMap(t, src, explicit)
		b := mustMap(t, src, zero)
		if diffRGB(a, b) > 1e-12 {
			t.Fatalf("零值拐点未使用默认 0.9: %v vs %v", a, b)
		}
	}
}

func diffRGB(a, b RGB) float64 {
	return math.Abs(a.R-b.R) + math.Abs(a.G-b.G) + math.Abs(a.B-b.B)
}

func TestGamutInputErrors(t *testing.T) {
	base := DefaultGamutConfig()

	cases := []struct {
		name      string
		mutate    func(*GamutConfig)
		wantErr   error
		wantField string
	}{
		{"gamma 为零", func(c *GamutConfig) { c.Gamma = 0 }, ErrGamma, "gamma"},
		{"gamma 为负", func(c *GamutConfig) { c.Gamma = -1 }, ErrGamma, "gamma"},
		{"gamma 为 NaN", func(c *GamutConfig) { c.Gamma = math.NaN() }, ErrGamma, "gamma"},
		{"gamma 为无穷", func(c *GamutConfig) { c.Gamma = math.Inf(1) }, ErrGamma, "gamma"},
		{"源矩阵奇异", func(c *GamutConfig) { c.SourceToXYZ = Mat3{} }, ErrSingular, "sourceMatrix"},
		{"目标矩阵奇异", func(c *GamutConfig) { c.TargetToXYZ = Mat3{} }, ErrSingular, "targetMatrix"},
		{"源矩阵含 NaN", func(c *GamutConfig) { c.SourceToXYZ[1][1] = math.NaN() }, ErrSingular, "sourceMatrix"},
		{"拐点越界", func(c *GamutConfig) { c.Knee = 1 }, ErrKnee, "knee"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			tc.mutate(&cfg)
			_, err := MapPixel(RGB{1, 0, 0}, cfg)
			if err == nil {
				t.Fatal("应报错")
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("错误类型不符: %v", err)
			}
			var fe *FieldError
			if !errors.As(err, &fe) {
				t.Fatalf("未返回字段错误: %v", err)
			}
			if fe.Field != tc.wantField {
				t.Fatalf("字段名 %q，期望 %q", fe.Field, tc.wantField)
			}
		})
	}

	t.Run("图为空", func(t *testing.T) {
		empties := []*ColorImage{
			nil,
			{},
			{W: 2, H: 2},
			{W: 2, H: 2, Pix: make([]RGB, 3)},
		}
		for i, im := range empties {
			_, err := MapToGamut(im, base)
			if err == nil {
				t.Fatalf("空图 #%d 应报错", i)
			}
			if !errors.Is(err, ErrEmptyColor) {
				t.Fatalf("空图 #%d 错误类型不符: %v", i, err)
			}
			var fe *FieldError
			if !errors.As(err, &fe) || fe.Field != "image" {
				t.Fatalf("空图 #%d 字段名不符: %v", i, err)
			}
		}
	})
}

// TestGamutReport 输出收尾所需的超域像素数与色差表。
func TestGamutReport(t *testing.T) {
	cfg := DefaultGamutConfig()
	m, err := newGamutMapper(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%-8s %-22s %-24s %-9s %-9s %-8s %-10s",
		"用例", "源 P3 编码", "目标 sRGB 编码", "映射前域外", "编码前残差", "色相°", "ΔE_OK")
	for _, tc := range gamutCases {
		out, residual := m.mapEncodedFull(tc.in)
		srcLab := m.labOfSource(tc.in)
		outLab := m.labOfTarget(out)
		t.Logf("%-8s (%.3f,%.3f,%.3f)  (%.4f,%.4f,%.4f)  %-9v %-9v %+.5f  %.5f",
			tc.name, tc.in.R, tc.in.G, tc.in.B, out.R, out.G, out.B,
			m.outOfGamut(tc.in), residual, hueShiftDeg(srcLab, outLab), deltaEOK(srcLab, outLab))
	}
	grid := buildGamutGrid(t)
	_, q, err := MeasureMap(grid, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("网格样本 %d 像素: 超域 %d (%.1f%%) -> 编码前残差 %d (%.1f%%)；最大色相偏移 %.5f°，最大 ΔE_OK %.5f",
		q.Total, q.BeforeOutOfGamut, 100*float64(q.BeforeOutOfGamut)/float64(q.Total),
		q.ResidualOutOfGamut, 100*float64(q.ResidualOutOfGamut)/float64(q.Total),
		q.MaxHueShiftDeg, q.MaxDeltaEOK)
}
