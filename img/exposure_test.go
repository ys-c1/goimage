package img_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"goimage/img"
	"goimage/internal/testimg"
)

// findPair 按名称取一对用例。
func findPair(t *testing.T, name string) testimg.Pair {
	t.Helper()
	for _, p := range testimg.All() {
		if p.Name == name {
			return p
		}
	}
	t.Fatalf("找不到用例 %s", name)
	return testimg.Pair{}
}

func fitOK(t *testing.T, p testimg.Pair) *img.ExposureFit {
	t.Helper()
	fit, err := img.FitExposure(p.A, p.B, p.Overlap)
	if err != nil {
		t.Fatalf("%s 拟合失败: %v", p.Name, err)
	}
	return fit
}

// TestExposureFitFiveCases 五组用例各跑一次，核对模型形式与已知曝光关系。
func TestExposureFitFiveCases(t *testing.T) {
	wantGain := map[string]float64{
		"同曝光": 1,
		"差一档": 0.5,
		"差三档": 0.125,
	}
	for _, p := range testimg.All() {
		fit := fitOK(t, p)
		if fit.Base != 0 || fit.Gain[0] != 1 || fit.Bias[0] != 0 {
			t.Fatalf("%s 基准图未固定为恒等: g=%v b=%v", p.Name, fit.Gain[0], fit.Bias[0])
		}
		if want, ok := wantGain[p.Name]; ok {
			if math.Abs(fit.Gain[1]-want) > 0.002 {
				t.Fatalf("%s 增益 %.6f，期望约 %.3f", p.Name, fit.Gain[1], want)
			}
		}
		if fit.Gain[1] < 0 {
			t.Fatalf("%s 增益为负: %.6f", p.Name, fit.Gain[1])
		}
	}
	// 纯色：重叠区无对比度，增益不可辨识，由正则项固定为 1，偏置吸收均值差。
	solid := fitOK(t, findPair(t, "纯色"))
	if !solid.Undetermined {
		t.Fatal("纯色用例应标记为欠定")
	}
	if math.Abs(solid.Gain[1]-1) > 1e-9 || math.Abs(solid.Bias[1]-(-102)) > 1e-9 {
		t.Fatalf("纯色正则解错误: g=%.9f b=%.9f", solid.Gain[1], solid.Bias[1])
	}
	if solid.Lambda <= 0 {
		t.Fatalf("纯色应使用正的正则强度，实际 %v", solid.Lambda)
	}
}

// TestExposureOverlapResidualReduced 核验一：补偿后重叠区亮度差至少缩小一个量级。
// 只统计两侧都未饱和的可用样本。
func TestExposureOverlapResidualReduced(t *testing.T) {
	for _, p := range testimg.All() {
		fit := fitOK(t, p)
		if fit.Usable == 0 {
			t.Fatalf("%s 无可用样本", p.Name)
		}
		if fit.ResidualBefore == 0 {
			if fit.ResidualAfter != 0 {
				t.Fatalf("%s 补偿前残差为零，补偿后为 %.6f", p.Name, fit.ResidualAfter)
			}
			continue
		}
		if fit.ResidualAfter*10 > fit.ResidualBefore {
			t.Fatalf("%s 重叠区残差未缩小一个量级: %.4f → %.4f",
				p.Name, fit.ResidualBefore, fit.ResidualAfter)
		}
	}
}

// TestExposureNonOverlapNoClip 核验二：不重叠区补偿后不得出现截断。
// 与核验一相互独立，只检查图 2 独有区的取值范围。
func TestExposureNonOverlapNoClip(t *testing.T) {
	for _, p := range testimg.All() {
		fit := fitOK(t, p)
		if fit.NonOverlapClip != 0 {
			t.Fatalf("%s 不重叠区出现 %d 个截断像素，范围 [%.4f, %.4f]",
				p.Name, fit.NonOverlapClip, fit.NonOverlapMin, fit.NonOverlapMax)
		}
		if fit.NonOverlapMin < 0 || fit.NonOverlapMax > 255 {
			t.Fatalf("%s 不重叠区范围越界 [%.4f, %.4f]",
				p.Name, fit.NonOverlapMin, fit.NonOverlapMax)
		}
	}
}

// TestExposureRejectsEmptyOverlap 重叠区为空一律拒绝并报出图号。
func TestExposureRejectsEmptyOverlap(t *testing.T) {
	p := findPair(t, "同曝光")
	_, err := img.FitExposure(p.A, p.B, 0)
	assertFigureError(t, err, img.ErrOverlapEmpty, "图 1|2")
}

// TestExposureRejectsNegativeGain 增益为负一律拒绝并报出图号（图 2）。
func TestExposureRejectsNegativeGain(t *testing.T) {
	a, _ := img.New(16, 4)
	b, _ := img.New(16, 4)
	// 重叠 4 列：a 递增、b 递减，最小二乘增益约为 -1。
	av := []uint8{10, 40, 80, 120}
	bv := []uint8{200, 170, 130, 90}
	for y := 0; y < 4; y++ {
		for k := 0; k < 4; k++ {
			a.Pix[y*16+(16-4+k)] = av[k]
			b.Pix[y*16+k] = bv[k]
		}
	}
	_, err := img.FitExposure(a, b, 4)
	assertFigureError(t, err, img.ErrNegativeGain, "图 2")
}

// TestExposureRejectsEmptyImage 空图一律拒绝并报出图号。
func TestExposureRejectsEmptyImage(t *testing.T) {
	p := findPair(t, "同曝光")
	_, err := img.FitExposure(nil, p.B, p.Overlap)
	assertFigureError(t, err, img.ErrEmptyImage, "图 1")
	_, err = img.FitExposure(p.A, nil, p.Overlap)
	assertFigureError(t, err, img.ErrEmptyImage, "图 2")
	_, err = img.FitExposure(p.A, &img.Gray{}, p.Overlap)
	assertFigureError(t, err, img.ErrEmptyImage, "图 2")
}

// TestStitchDirectUnchanged 直通拼接不做补偿：输出与两图原始像素逐位一致。
func TestStitchDirectUnchanged(t *testing.T) {
	for _, p := range testimg.All() {
		out, err := img.StitchDirect(p.A, p.B, p.Overlap)
		if err != nil {
			t.Fatalf("%s 直通拼接失败: %v", p.Name, err)
		}
		if out.W != testimg.SceneW || out.H != testimg.PairH {
			t.Fatalf("%s 画幅 %dx%d", p.Name, out.W, out.H)
		}
		want, err := img.New(testimg.SceneW, testimg.PairH)
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < p.A.H; y++ {
			copy(want.Pix[y*want.W:y*want.W+p.A.W], p.A.Pix[y*p.A.W:y*p.A.W+p.A.W])
			for x := p.Overlap; x < p.B.W; x++ {
				want.Pix[y*want.W+p.A.W+(x-p.Overlap)] = p.B.Pix[y*p.B.W+x]
			}
		}
		for i := range want.Pix {
			if out.Pix[i] != want.Pix[i] {
				t.Fatalf("%s 直通拼接第 %d 像素被改动: %d ≠ %d",
					p.Name, i, out.Pix[i], want.Pix[i])
			}
		}
		// 固定尺寸再确认一次。
		if out.W != 112 || out.H != 32 {
			t.Fatalf("%s 直通输出画幅 %dx%d，期望 112x32", p.Name, out.W, out.H)
		}
	}
}

// TestStitchExposure 补偿拼接：重叠区用补偿后的图 2，局部过曝处回退到图 1，
// 最终输出不残留饱和块。
func TestStitchExposure(t *testing.T) {
	for _, p := range testimg.All() {
		out, fit, err := img.StitchExposure(p.A, p.B, p.Overlap)
		if err != nil {
			t.Fatalf("%s 补偿拼接失败: %v", p.Name, err)
		}
		if out.W != testimg.SceneW || out.H != testimg.PairH {
			t.Fatalf("%s 画幅 %dx%d", p.Name, out.W, out.H)
		}
		// 图 1 独有区（左侧）逐位保留。
		for y := 0; y < p.A.H; y++ {
			for x := 0; x < p.A.W-p.Overlap; x++ {
				if out.Pix[y*out.W+x] != p.A.Pix[y*p.A.W+x] {
					t.Fatalf("%s 图 1 独有区像素被改动 (%d,%d)", p.Name, x, y)
				}
			}
		}
		// 重叠区：未饱和处应与图 1 接近，且不得为 255 的饱和残留（图 1 未饱和时）。
		for y := 0; y < p.A.H; y++ {
			for k := 0; k < p.Overlap; k++ {
				av := p.A.Pix[y*p.A.W+(p.A.W-p.Overlap+k)]
				bv := p.B.Pix[y*p.B.W+k]
				got := out.Pix[y*out.W+(p.A.W-p.Overlap+k)]
				if bv == 255 && av != 255 && got != av {
					t.Fatalf("%s 重叠区饱和像素未回退到图 1 (%d,%d): %d ≠ %d",
						p.Name, k, y, got, av)
				}
			}
		}
		if fit.FallbackToBase != 0 && p.Name != "局部过曝" {
			t.Fatalf("%s 不应有回退像素，实际 %d", p.Name, fit.FallbackToBase)
		}
	}
	// 局部过曝：96 个饱和像素全部回退。
	p := findPair(t, "局部过曝")
	_, fit := mustStitch(t, p)
	if fit.FallbackToBase != fit.Saturated || fit.FallbackToBase != 96 {
		t.Fatalf("局部过曝回退 %d、饱和 %d，期望均为 96", fit.FallbackToBase, fit.Saturated)
	}
}

func mustStitch(t *testing.T, p testimg.Pair) (*img.Gray, *img.ExposureFit) {
	t.Helper()
	out, fit, err := img.StitchExposure(p.A, p.B, p.Overlap)
	if err != nil {
		t.Fatalf("%s 补偿拼接失败: %v", p.Name, err)
	}
	return out, fit
}

// assertFigureError 断言错误可归一到指定哨兵错误，且错误信息带指定图号。
func assertFigureError(t *testing.T, err error, want error, figures string) {
	t.Helper()
	if err == nil {
		t.Fatalf("期望拒绝（%v），实际未报错", want)
	}
	if !errors.Is(err, want) {
		t.Fatalf("错误类型错误: %v，期望 %v", err, want)
	}
	var fe *img.FigureError
	if !errors.As(err, &fe) {
		t.Fatalf("错误未携带图号: %v", err)
	}
	if fe.Figures != figures {
		t.Fatalf("图号 %q，期望 %q（错误：%v）", fe.Figures, figures, err)
	}
	if !strings.Contains(err.Error(), figures) {
		t.Fatalf("错误信息未包含图号 %q: %v", figures, err)
	}
}
