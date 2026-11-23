package img

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

func solidImg(w, h int, v uint8) *Gray {
	g, _ := New(w, h)
	for i := range g.Pix {
		g.Pix[i] = v
	}
	return g
}

func stepEdgeImg(w, h, xEdge int, lo, hi uint8) *Gray {
	g, _ := New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := lo
			if x >= xEdge {
				v = hi
			}
			g.Pix[y*w+x] = v
		}
	}
	return g
}

func thinLineImg(w, h, xLine int, bg, fg uint8) *Gray {
	g := solidImg(w, h, bg)
	for y := 0; y < h; y++ {
		g.Pix[y*w+xLine] = fg
	}
	return g
}

func textureImg(w, h int, lo, hi uint8) *Gray {
	g, _ := New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := lo
			if (x+y)%2 == 0 {
				v = hi
			}
			g.Pix[y*w+x] = v
		}
	}
	return g
}

func noisyFlatImg(w, h int, base uint8, sigma float64, seed int64) *Gray {
	g := solidImg(w, h, base)
	rng := rand.New(rand.NewSource(seed))
	for i := range g.Pix {
		v := float64(base) + rng.NormFloat64()*sigma
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		g.Pix[i] = uint8(math.Round(v))
	}
	return g
}

func TestBilateralRejectsBadParams(t *testing.T) {
	g := solidImg(8, 8, 100)
	cases := []struct {
		name  string
		img   *Gray
		p     BilateralParams
		field string
	}{
		{"半径为零", g, BilateralParams{Radius: 0, SigmaS: 1, SigmaR: 20}, "Radius"},
		{"半径为负", g, BilateralParams{Radius: -1, SigmaS: 1, SigmaR: 20}, "Radius"},
		{"空间sigma为零", g, BilateralParams{Radius: 2, SigmaS: 0, SigmaR: 20}, "SigmaS"},
		{"空间sigma为负", g, BilateralParams{Radius: 2, SigmaS: -1, SigmaR: 20}, "SigmaS"},
		{"灰度sigma为NaN", g, BilateralParams{Radius: 2, SigmaS: 1, SigmaR: math.NaN()}, "SigmaR"},
		{"灰度sigma为Inf", g, BilateralParams{Radius: 2, SigmaS: 1, SigmaR: math.Inf(1)}, "SigmaR"},
		{"空图", &Gray{}, BilateralParams{Radius: 2, SigmaS: 1, SigmaR: 20}, "Image"},
		{"nil图", nil, BilateralParams{Radius: 2, SigmaS: 1, SigmaR: 20}, "Image"},
		{"半径超反解上限", g, BilateralParams{Radius: 3, SigmaS: 1, SigmaR: 20, NoiseReduction: 4}, "Radius"},
	}
	for _, c := range cases {
		_, err := Bilateral(c.img, c.p)
		if err == nil {
			t.Fatalf("%s: 应报错", c.name)
		}
		if !strings.Contains(err.Error(), c.field) {
			t.Fatalf("%s: 错误应含字段名 %s，实际 %v", c.name, c.field, err)
		}
	}
}

func TestSolveSigmaR(t *testing.T) {
	s, steps, err := SolveSigmaR(100, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	want := 100 / math.Sqrt(2*math.Log(1.8/0.2))
	if math.Abs(s-want) > 1e-9 {
		t.Fatalf("sigma 反解错误: got %v want %v", s, want)
	}
	if len(steps) == 0 {
		t.Fatal("反解过程为空")
	}
	if _, _, err := SolveSigmaR(0, 0.8); err == nil {
		t.Fatal("边缘反差为零应报错")
	}
	if _, _, err := SolveSigmaR(100, 1.2); err == nil {
		t.Fatal("保留率越界应报错")
	}
}

func TestSolveRadius(t *testing.T) {
	r, steps, err := SolveRadius(4)
	if err != nil {
		t.Fatal(err)
	}
	want := int(math.Ceil(math.Sqrt(4 / math.Pi)))
	if r != want {
		t.Fatalf("半径反解错误: got %d want %d", r, want)
	}
	if len(steps) == 0 {
		t.Fatal("反解过程为空")
	}
	if _, _, err := SolveRadius(1); err == nil {
		t.Fatal("抑制倍数不大于一应报错")
	}
}

func TestStepEdgePreserved(t *testing.T) {
	g := stepEdgeImg(64, 64, 32, 50, 150) // 边缘反差 h=100
	_, rep, err := Denoise(g, 100, 0.8, 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgeRetention < 0.8 {
		t.Fatalf("阶跃边保留率不足: %.4f < 0.8", rep.EdgeRetention)
	}
	t.Logf("阶跃边: 保留率=%.4f σr=%.4g R=%d", rep.EdgeRetention, rep.SigmaR, rep.Radius)
}

func TestThinLinePreserved(t *testing.T) {
	g := thinLineImg(64, 64, 32, 50, 200) // 细线反差 h=150
	_, rep, err := Denoise(g, 150, 0.8, 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgeRetention < 0.8 {
		t.Fatalf("细线保留率不足: %.4f < 0.8", rep.EdgeRetention)
	}
	t.Logf("细线: 保留率=%.4f σr=%.4g R=%d", rep.EdgeRetention, rep.SigmaR, rep.Radius)
}

func TestTexturePreserved(t *testing.T) {
	g := textureImg(64, 64, 90, 150) // 纹理反差 h=60
	_, rep, err := Denoise(g, 60, 0.7, 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgeRetention < 0.7 {
		t.Fatalf("纹理保留率不足: %.4f < 0.7", rep.EdgeRetention)
	}
	t.Logf("纹理: 保留率=%.4f σr=%.4g R=%d", rep.EdgeRetention, rep.SigmaR, rep.Radius)
}

func TestNoiseSuppressed(t *testing.T) {
	g := noisyFlatImg(64, 64, 128, 10, 42)
	_, rep, err := Denoise(g, 60, 0.8, 4, []int{4, 4, 60, 60})
	if err != nil {
		t.Fatal(err)
	}
	if rep.VarianceDrop < 4 {
		t.Fatalf("平坦区方差降幅不足: %.4f < 4", rep.VarianceDrop)
	}
	t.Logf("噪点: 方差降幅=%.4f σr=%.4g R=%d", rep.VarianceDrop, rep.SigmaR, rep.Radius)
}

func TestSolidColorUnchanged(t *testing.T) {
	g := solidImg(32, 32, 200)
	out, rep, err := Denoise(g, 50, 0.8, 4, []int{0, 0, 32, 32})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out.Pix {
		if v != g.Pix[i] {
			t.Fatalf("纯色图第 %d 像素被改动: %d -> %d", i, g.Pix[i], v)
		}
	}
	if rep.EdgeRetention != 1 {
		t.Fatalf("纯色图保留率应为 1，实际 %.4f", rep.EdgeRetention)
	}
	t.Logf("纯色: 保留率=%.4f 方差降幅=%.4f", rep.EdgeRetention, rep.VarianceDrop)
}

func TestDenoiseReportOnNoisyEdge(t *testing.T) {
	g := stepEdgeImg(64, 64, 32, 50, 150)
	rng := rand.New(rand.NewSource(7))
	for i := range g.Pix {
		v := float64(g.Pix[i]) + rng.NormFloat64()*8
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		g.Pix[i] = uint8(math.Round(v))
	}
	_, rep, err := Denoise(g, 100, 0.8, 4, []int{2, 2, 24, 62})
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgeRetention < 0.8 {
		t.Fatalf("含噪阶跃边保留率不足: %.4f", rep.EdgeRetention)
	}
	if rep.VarianceDrop < 4 {
		t.Fatalf("含噪阶跃边方差降幅不足: %.4f", rep.VarianceDrop)
	}
	if len(rep.Derivation) == 0 {
		t.Fatal("报告缺少反解过程")
	}
	t.Logf("含噪阶跃边: 保留率=%.4f 方差降幅=%.4f σr=%.4g R=%d σs=%.4g",
		rep.EdgeRetention, rep.VarianceDrop, rep.SigmaR, rep.Radius, rep.SigmaS)
	for _, s := range rep.Derivation {
		t.Logf("反解: %s", s)
	}
}
