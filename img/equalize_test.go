package img

import (
	"errors"
	"testing"
)

func darkImg() *Gray {
	g, _ := New(32, 32)
	for i := range g.Pix {
		g.Pix[i] = uint8((i * 7) % 41) // 0..40
	}
	return g
}

func brightImg() *Gray {
	g, _ := New(32, 32)
	for i := range g.Pix {
		g.Pix[i] = uint8(215 + (i*3)%41) // 215..255
	}
	return g
}

func bimodalImg() *Gray {
	g, _ := New(32, 32)
	for i := range g.Pix {
		if i%4 < 3 {
			g.Pix[i] = uint8((i * 5) % 30) // 暗簇：窄而高
		} else {
			g.Pix[i] = uint8(190 + (i*7)%60) // 亮簇：宽而矮
		}
	}
	return g
}

func solidImg() *Gray {
	g, _ := New(8, 8)
	for i := range g.Pix {
		g.Pix[i] = 128
	}
	return g
}

func singlePixelImg() *Gray {
	g, _ := New(1, 1)
	g.Pix[0] = 200
	return g
}

func fixtures() map[string]*Gray {
	return map[string]*Gray{
		"暗图":  darkImg(),
		"亮图":  brightImg(),
		"双峰":  bimodalImg(),
		"纯色":  solidImg(),
		"单像素": singlePixelImg(),
	}
}

func pixSum(p []uint8) uint64 {
	var s uint64
	for _, v := range p {
		s += uint64(v)
	}
	return s
}

func TestEqualizeAreaConserved(t *testing.T) {
	for name, g := range fixtures() {
		out, rep, err := Equalize(g, EqualizeOpts{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		n := g.W * g.H
		if rep.HistAreaBefore != n || rep.HistAreaAfter != n {
			t.Fatalf("%s: 直方图面积不守恒 前=%d 后=%d 应=%d", name, rep.HistAreaBefore, rep.HistAreaAfter, n)
		}
		if rep.PixSumBefore != pixSum(g.Pix) {
			t.Fatalf("%s: 前像素总和证据不符 %d != %d", name, rep.PixSumBefore, pixSum(g.Pix))
		}
		if rep.PixSumAfter != pixSum(out.Pix) {
			t.Fatalf("%s: 后像素总和证据不符 %d != %d", name, rep.PixSumAfter, pixSum(out.Pix))
		}
	}
}

func TestEqualizePassthrough(t *testing.T) {
	g := bimodalImg()
	out, rep, err := Equalize(g, EqualizeOpts{Passthrough: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Pix {
		if out.Pix[i] != g.Pix[i] {
			t.Fatalf("直通输出第 %d 像素被改动", i)
		}
	}
	if rep.PixSumAfter != rep.PixSumBefore {
		t.Fatal("直通前后像素总和应相等")
	}
	back, err := Inverse(out, rep)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Pix {
		if back.Pix[i] != g.Pix[i] {
			t.Fatalf("直通往返第 %d 像素不一致", i)
		}
	}
}

func TestEqualizeFieldErrors(t *testing.T) {
	cases := []struct {
		name  string
		call  func() error
		field string
	}{
		{"图为空", func() error { _, _, err := Equalize(&Gray{}, EqualizeOpts{}); return err }, "Pix"},
		{"裁剪过小", func() error {
			_, _, err := Equalize(darkImg(), EqualizeOpts{ClipRatio: 1e-9})
			return err
		}, "ClipRatio"},
		{"裁剪过大", func() error {
			_, _, err := Equalize(darkImg(), EqualizeOpts{ClipRatio: 1e9})
			return err
		}, "ClipRatio"},
		{"色阶非法", func() error {
			_, _, err := Equalize(darkImg(), EqualizeOpts{Levels: 257})
			return err
		}, "Levels"},
	}
	for _, c := range cases {
		err := c.call()
		var fe *FieldError
		if !errors.As(err, &fe) {
			t.Fatalf("%s: 应返回 FieldError，得到 %v", c.name, err)
		}
		if fe.Field != c.field {
			t.Fatalf("%s: 字段名应为 %s，得到 %s", c.name, c.field, fe.Field)
		}
	}
}

func TestRoundTripBound(t *testing.T) {
	for name, g := range fixtures() {
		out, rep, err := Equalize(g, EqualizeOpts{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		back, err := Inverse(out, rep)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		exact := 0
		for i := range g.Pix {
			d := int(g.Pix[i]) - int(back.Pix[i])
			if d < 0 {
				d = -d
			}
			if d > exact {
				exact = d
			}
			if d > rep.RoundTripBound {
				t.Fatalf("%s: 第 %d 像素往返误差 %d 超过上界 %d", name, i, d, rep.RoundTripBound)
			}
		}
		if exact != rep.RoundTripMaxError {
			t.Fatalf("%s: 报告往返误差 %d 与实际 %d 不符", name, rep.RoundTripMaxError, exact)
		}
		if rep.RoundTripMaxError > rep.RoundTripBound {
			t.Fatalf("%s: 往返误差 %d 超过上界 %d", name, rep.RoundTripMaxError, rep.RoundTripBound)
		}
	}
}

func TestAutoClipGainMaximal(t *testing.T) {
	ratios := []float64{0, 1, 2, 4, 8, 32}
	for _, name := range []string{"暗图", "亮图", "双峰"} {
		g := fixtures()[name]
		rows, err := GainTable(g, ratios)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		auto := rows[0].Gain
		for _, r := range rows[1:] {
			if r.Gain > auto+1e-9 {
				t.Fatalf("%s: 比例 %g 增益 %f 超过解析最优 %f", name, r.Ratio, r.Gain, auto)
			}
		}
	}
}

func TestEqualizeSpreadsDark(t *testing.T) {
	g := darkImg()
	out, _, err := Equalize(g, EqualizeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	span := func(p []uint8) int {
		lo, hi := 255, 0
		for _, v := range p {
			if int(v) < lo {
				lo = int(v)
			}
			if int(v) > hi {
				hi = int(v)
			}
		}
		return hi - lo
	}
	if span(out.Pix) <= span(g.Pix) {
		t.Fatalf("暗图均衡后动态范围应拉大：前 %d 后 %d", span(g.Pix), span(out.Pix))
	}
}

func TestSolidUniform(t *testing.T) {
	out, _, err := Equalize(solidImg(), EqualizeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out.Pix {
		if v != out.Pix[0] {
			t.Fatalf("纯色图均衡后应仍均一，第 %d 像素 %d != %d", i, v, out.Pix[0])
		}
	}
}

func TestSinglePixel(t *testing.T) {
	out, rep, err := Equalize(singlePixelImg(), EqualizeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Pix) != 1 {
		t.Fatal("单像素输出尺寸错误")
	}
	if rep.HistAreaBefore != 1 || rep.HistAreaAfter != 1 {
		t.Fatal("单像素直方图面积应为 1")
	}
}

func TestGainTablePrint(t *testing.T) {
	for _, name := range []string{"暗图", "亮图", "双峰"} {
		g := fixtures()[name]
		rows, err := GainTable(g, []float64{0, 1, 2, 4, 8, 32})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range rows {
			tag := "     "
			if r.Ratio == 0 {
				tag = "(最优)"
			}
			t.Logf("%s 比例=%5.1f%s 裁剪限=%4d 增益=%.4f", name, r.Ratio, tag, r.ClipLimit, r.Gain)
		}
	}
}
