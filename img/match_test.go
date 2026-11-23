package img

import (
	"errors"
	"math"
	"testing"
)

// 目标分布：0.7 权重指数衰减（峰在 0 附近）+ 0.3 权重高斯（峰在 220），
// 双峰且偏斜，均值与中位数分离，便于区分各方法。
func targetDist() []float64 {
	t := make([]float64, grayLevels)
	for v := 0; v < grayLevels; v++ {
		x := float64(v)
		t[v] = 0.7*math.Exp(-x/25) + 0.3*math.Exp(-(x-220)*(x-220)/(2*15*15))
	}
	return t
}

func gaussHist(mu, sigma, weight float64, h *[grayLevels]float64) {
	for v := 0; v < grayLevels; v++ {
		d := (float64(v) - mu) / sigma
		h[v] += weight * math.Exp(-d*d/2)
	}
}

// makeImage 按直方图形状确定性生成 n 个像素的图。
func makeImage(h *[grayLevels]float64, n int) *Gray {
	g, err := New(n, 1)
	if err != nil {
		panic(err)
	}
	sum := 0.0
	for _, v := range h {
		sum += v
	}
	idx := 0
	for v := 0; v < grayLevels; v++ {
		c := int(math.Round(h[v] / sum * float64(n)))
		for ; c > 0 && idx < n; c-- {
			g.Pix[idx] = uint8(v)
			idx++
		}
	}
	for idx < n { // 舍入差额补到最大级
		g.Pix[idx] = grayLevels - 1
		idx++
	}
	return g
}

const testBins = 64

func runCase(t *testing.T, name string, g *Gray) *MatchResult {
	t.Helper()
	r, err := MatchHistogram(g, targetDist(), testBins)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	// 映射单调不减
	for v := 0; v < grayLevels-1; v++ {
		if r.LUT[v] > r.LUT[v+1] {
			t.Fatalf("%s: LUT 在 %d 处下降", name, v)
		}
	}
	if !r.Converged {
		t.Fatalf("%s: Lloyd 未收敛", name)
	}
	if r.Iterations < 1 {
		t.Fatalf("%s: 迭代次数异常 %d", name, r.Iterations)
	}
	if r.W1 > r.W1Bound+1e-9 {
		t.Fatalf("%s: W1=%.4f 超过上界 %.4f", name, r.W1, r.W1Bound)
	}
	if !(r.W1 < r.UniformW1) {
		t.Fatalf("%s: 均匀分箱基线未输 %.4f vs %.4f", name, r.W1, r.UniformW1)
	}
	if !(r.W1 < r.LinearW1) {
		t.Fatalf("%s: 线性映射基线未输 %.4f vs %.4f", name, r.W1, r.LinearW1)
	}
	t.Logf("%s: 迭代=%d W1=%.4f 上界=%.4f 均匀分箱=%.4f 线性映射=%.4f",
		name, r.Iterations, r.W1, r.W1Bound, r.UniformW1, r.LinearW1)
	return r
}

func TestMatchUnimodal(t *testing.T) {
	var h [grayLevels]float64
	gaussHist(80, 15, 1, &h)
	runCase(t, "单峰", makeImage(&h, 8192))
}

func TestMatchBimodal(t *testing.T) {
	var h [grayLevels]float64
	gaussHist(50, 10, 0.5, &h)
	gaussHist(200, 12, 0.5, &h)
	runCase(t, "双峰", makeImage(&h, 8192))
}

func TestMatchUniform(t *testing.T) {
	var h [grayLevels]float64
	for v := range h {
		h[v] = 1
	}
	runCase(t, "均匀", makeImage(&h, 8192))
}

func TestMatchExtremeSkew(t *testing.T) {
	var h [grayLevels]float64
	h[5] = 0.95
	for v := 100; v < grayLevels; v++ {
		h[v] = 0.05 / 156
	}
	runCase(t, "极端偏斜", makeImage(&h, 8192))
}

func TestMatchSinglePixel(t *testing.T) {
	g, err := New(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	g.Pix[0] = 150
	runCase(t, "单像素", g)
}

func TestMatchFieldErrors(t *testing.T) {
	var fe *FieldError

	if _, err := MatchHistogram(nil, targetDist(), testBins); !errors.As(err, &fe) || fe.Field != "Image" {
		t.Fatalf("空图应报 Image 字段, got %v", err)
	}
	g, _ := New(2, 2)
	if _, err := MatchHistogram(g, targetDist(), 0); !errors.As(err, &fe) || fe.Field != "Bins" {
		t.Fatalf("分箱为零应报 Bins 字段, got %v", err)
	}
	bad := targetDist()
	bad[3] = -1
	if _, err := MatchHistogram(g, bad, testBins); !errors.As(err, &fe) || fe.Field != "Target" {
		t.Fatalf("负值分布应报 Target 字段, got %v", err)
	}
	bad = targetDist()
	bad[7] = math.NaN()
	if _, err := MatchHistogram(g, bad, testBins); !errors.As(err, &fe) || fe.Field != "Target" {
		t.Fatalf("NaN 分布应报 Target 字段, got %v", err)
	}
	if _, err := MatchHistogram(g, make([]float64, grayLevels), testBins); !errors.As(err, &fe) || fe.Field != "Target" {
		t.Fatalf("零和分布应报 Target 字段, got %v", err)
	}
	if _, err := MatchHistogram(g, make([]float64, 10), testBins); !errors.As(err, &fe) || fe.Field != "Target" {
		t.Fatalf("长度非法应报 Target 字段, got %v", err)
	}
}
