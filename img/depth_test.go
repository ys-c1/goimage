package img

import (
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"
)

func baseOptions() CompressOptions {
	return CompressOptions{
		SrcDepth:   16,
		DstDepth:   8,
		LoQuantile: 0.01,
		HiQuantile: 0.99,
		Levels:     8,
		MinSpacing: 12,
		MaxSpacing: 24,
		Kernel:     KernelFloydSteinberg,
	}
}

func gradient16(w, h int) *Gray16 {
	g, _ := New16(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g.Pix[y*w+x] = uint16(uint32(x) * 65535 / uint32(w-1))
		}
	}
	return g
}

func photo16(w, h int) *Gray16 {
	r := rand.New(rand.NewSource(42))
	g, _ := New16(w, h)
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

func icon16(w, h int) *Gray16 {
	g, _ := New16(w, h)
	cx, cy, rad := float64(w)/2, float64(h)/2, float64(w)/4
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint16(6553) // 底 10%
			dx, dy := float64(x)-cx, float64(y)-cy
			if dx*dx+dy*dy < rad*rad {
				v = 58981 // 圆 90%
			}
			if x > w/2 && y > h/2 {
				v = 36044 // 块 55%
			}
			g.Pix[y*w+x] = v
		}
	}
	return g
}

func TestCompressGradient(t *testing.T) {
	out, rep, err := Compress16(gradient16(256, 64), baseOptions())
	if err != nil {
		t.Fatal(err)
	}
	if rep.LevelsUsed < 2 {
		t.Fatalf("渐变色阶过少 %d", rep.LevelsUsed)
	}
	if rep.Spacing < 12 || rep.Spacing > 24 {
		t.Fatalf("间距 %.2f 未落入 [12,24]", rep.Spacing)
	}
	if rep.Strength <= 0 || rep.Strength >= 1 {
		t.Fatalf("反解强度异常 %.4f", rep.Strength)
	}
	if LevelsUsed(out) != rep.LevelsUsed {
		t.Fatal("回执色阶数与输出不一致")
	}
}

func TestCompressPhoto(t *testing.T) {
	_, rep, err := Compress16(photo16(128, 128), baseOptions())
	if err != nil {
		t.Fatal(err)
	}
	if rep.LevelsUsed < 4 {
		t.Fatalf("照片色阶过少 %d", rep.LevelsUsed)
	}
	if rep.ClipHi <= rep.ClipLo {
		t.Fatalf("裁剪点异常 %d %d", rep.ClipLo, rep.ClipHi)
	}
}

func TestCompressIcon(t *testing.T) {
	_, rep, err := Compress16(icon16(64, 64), baseOptions())
	if err != nil {
		t.Fatal(err)
	}
	if rep.LevelsUsed > 8 {
		t.Fatalf("图标色阶应很少，实际 %d", rep.LevelsUsed)
	}
}

func TestCompressSolid(t *testing.T) {
	g, _ := New16(32, 32)
	for i := range g.Pix {
		g.Pix[i] = 30000
	}
	out, rep, err := Compress16(g, baseOptions())
	if err != nil {
		t.Fatal(err)
	}
	if rep.LevelsUsed != 1 {
		t.Fatalf("纯色应只剩一阶，实际 %d", rep.LevelsUsed)
	}
	for _, v := range out.Pix {
		if v != out.Pix[0] {
			t.Fatal("纯色输出不均匀")
		}
	}
}

func TestCompressSinglePixel(t *testing.T) {
	g, _ := New16(1, 1)
	g.Pix[0] = 40000
	_, rep, err := Compress16(g, baseOptions())
	if err != nil {
		t.Fatal(err)
	}
	if rep.LevelsUsed != 1 {
		t.Fatalf("单像素应只有一阶，实际 %d", rep.LevelsUsed)
	}
}

func TestCompressFieldErrors(t *testing.T) {
	g := gradient16(8, 8)
	cases := []struct {
		name  string
		mut   func(*CompressOptions)
		img   *Gray16
		field string
		msg   string
	}{
		{"位深越界-源", func(o *CompressOptions) { o.SrcDepth = 24 }, g, "SrcDepth", "位深越界"},
		{"位深越界-目标", func(o *CompressOptions) { o.DstDepth = 16 }, g, "DstDepth", "位深越界"},
		{"分位倒置", func(o *CompressOptions) { o.LoQuantile, o.HiQuantile = 0.9, 0.1 }, g, "LoQuantile", "分位倒置"},
		{"图为空", func(o *CompressOptions) {}, &Gray16{}, "Image", "图为空"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			o := baseOptions()
			c.mut(&o)
			_, _, err := Compress16(c.img, o)
			if err == nil {
				t.Fatal("应报错")
			}
			var fe *FieldError
			if !errors.As(err, &fe) {
				t.Fatalf("应为 FieldError，实际 %T", err)
			}
			if fe.Field != c.field {
				t.Fatalf("字段名应为 %s，实际 %s", c.field, fe.Field)
			}
			if !strings.Contains(err.Error(), c.msg) {
				t.Fatalf("报错应含 %q：%v", c.msg, err)
			}
		})
	}
}

func TestCompressKernelConfigurable(t *testing.T) {
	for _, k := range []Kernel{KernelNone, KernelFloydSteinberg, KernelAtkinson} {
		o := baseOptions()
		o.Kernel = k
		if _, _, err := Compress16(gradient16(64, 16), o); err != nil {
			t.Fatalf("核 %s: %v", k.Name, err)
		}
	}
}

func TestCompressDepthConfigurable(t *testing.T) {
	o := baseOptions()
	o.SrcDepth = 12
	o.DstDepth = 4
	g, _ := New16(16, 16)
	for i := range g.Pix {
		g.Pix[i] = uint16(i * 16 % 4096)
	}
	if _, _, err := Compress16(g, o); err != nil {
		t.Fatal(err)
	}
}

func TestPassthrough8Unchanged(t *testing.T) {
	// 八位直通：Quantize 的既有语义由 TestQuantizeLevels 锁定，这里确认接口未变
	g, _ := New(4, 1)
	g.Pix = []uint8{0, 60, 130, 250}
	out, err := Quantize(g, 4)
	if err != nil {
		t.Fatal(err)
	}
	if LevelsUsed(out) > 4 {
		t.Fatal("直通路径色阶超限")
	}
}
