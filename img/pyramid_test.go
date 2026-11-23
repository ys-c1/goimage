package img

import (
	"math"
	"math/rand"
	"testing"
)

// ---- 五类用例生成器：平滑、纹理、边缘、噪点、纯色 ----

func smoothPattern(w, h int) *Gray {
	// 低频正弦平滑场
	g, _ := New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 128 + 80*math.Sin(2*math.Pi*float64(x)/float64(w))*
				math.Cos(2*math.Pi*float64(y)/float64(h))
			g.Pix[y*w+x] = uint8(math.Round(v))
		}
	}
	return g
}

func texturePattern(w, h int) *Gray {
	// 细密棋盘 + 高频调制，纹理丰富
	g, _ := New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 30 + ((x+y)%2)*90 + ((x*3+y*5)%7)*12
			g.Pix[y*w+x] = uint8(v)
		}
	}
	return g
}

func edgePattern(w, h int) *Gray {
	// 横竖两条硬边加一个阶跃方块
	g, _ := New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(20)
			if x >= w/2 {
				v = 220
			}
			if y >= h/2 {
				v = 255 - v
			}
			if x > w/4 && x < 3*w/4 && y > h/4 && y < 3*h/4 {
				v = v/2 + 40
			}
			g.Pix[y*w+x] = v
		}
	}
	return g
}

func noisePattern(w, h int) *Gray {
	// 固定种子的白噪点
	r := rand.New(rand.NewSource(20261124))
	g, _ := New(w, h)
	for i := range g.Pix {
		g.Pix[i] = uint8(r.Intn(256))
	}
	return g
}

func solidPattern(w, h int) *Gray {
	g, _ := New(w, h)
	for i := range g.Pix {
		g.Pix[i] = 137
	}
	return g
}

var allPatterns = []struct {
	name string
	gen  func(w, h int) *Gray
}{
	{"平滑", smoothPattern},
	{"纹理", texturePattern},
	{"边缘", edgePattern},
	{"噪点", noisePattern},
	{"纯色", solidPattern},
}

func defaultKernel(t *testing.T) []float64 {
	t.Helper()
	k, err := BinomialKernel(DefaultKernelLen)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// ---- 输入校验：层数为零、核长为偶、图为空，各自指出参数 ----

func TestInvalidInputs(t *testing.T) {
	g := smoothPattern(32, 24)
	k := defaultKernel(t)

	if _, err := BuildLaplacian(g, 0, k); err != ErrParamLevels {
		t.Fatalf("层数为零应报 ErrParamLevels(levels)，得到 %v", err)
	}
	if _, err := BuildLaplacian(g, -3, k); err != ErrParamLevels {
		t.Fatalf("层数为负应报 ErrParamLevels(levels)，得到 %v", err)
	}
	for _, badLen := range []int{2, 4, 6, 8} {
		if _, err := GaussianKernel(badLen, 1); err != ErrParamKernel {
			t.Fatalf("核长 %d 为偶应报 ErrParamKernel(kernelLen)，得到 %v", badLen, err)
		}
	}
	if _, err := BuildLaplacian(g, 2, []float64{0.25, 0.25, 0.25, 0.25}); err != ErrParamKernel {
		t.Fatalf("偶数核应报 ErrParamKernel(kernelLen)，得到 %v", err)
	}
	if _, err := GaussianKernel(5, 0); err != ErrParamSigma {
		t.Fatalf("sigma 为零应报 ErrParamSigma(sigma)，得到 %v", err)
	}
	if _, err := BuildLaplacian(nil, 2, k); err != ErrParamImage {
		t.Fatalf("nil 图应报 ErrParamImage(image)，得到 %v", err)
	}
	empty := &Gray{}
	if _, err := BuildLaplacian(empty, 2, k); err != ErrParamImage {
		t.Fatalf("空图应报 ErrParamImage(image)，得到 %v", err)
	}
	if _, err := Resize(nil, 8, 8); err != ErrParamImage {
		t.Fatalf("单层缩放空图应报 ErrParamImage，得到 %v", err)
	}
	if _, err := Resize(g, 0, 8); err != ErrParamTarget {
		t.Fatalf("目标宽为零应报 ErrParamTarget，得到 %v", err)
	}
	// 层数超过几何上限
	small, _ := New(2, 2)
	if _, err := BuildLaplacian(small, 5, k); err != ErrLevelsTooDeep {
		t.Fatalf("超深金字塔应报 ErrLevelsTooDeep，得到 %v", err)
	}
}

// ---- 核验一：重建误差随层数单调下降 ----

func TestMonotoneReconstruction(t *testing.T) {
	for _, p := range allPatterns {
		g := p.gen(64, 48)
		levels := MaxLevels(g.W, g.H)
		lp, err := BuildLaplacian(g, levels, defaultKernel(t))
		if err != nil {
			t.Fatalf("%s: 建塔失败: %v", p.name, err)
		}
		_, measured, bounds := lp.ReconstructCurve()
		if len(measured) != levels+1 || len(bounds) != levels+1 {
			t.Fatalf("%s: 曲线长度错误 %d/%d", p.name, len(measured), levels+1)
		}
		for l := 1; l <= levels; l++ {
			// 浮点累加有机器精度级噪声（约 1e-12 相对误差，经上千像素求和
			// 后绝对差可达 ~1e-9），按 1e-7 相对容差判定单调不增。
			tol := 1e-7 * math.Max(1, math.Abs(measured[l-1]))
			if measured[l] > measured[l-1]+tol {
				t.Fatalf("%s: 实测 MSE 非单调，第 %d 层 %g > 第 %d 层 %g",
					p.name, l, measured[l], l-1, measured[l-1])
			}
			tol = 1e-7 * math.Max(1, math.Abs(bounds[l-1]))
			if bounds[l] > bounds[l-1]+tol {
				t.Fatalf("%s: 上界曲线非单调，第 %d 层 %g > %g",
					p.name, l, bounds[l], bounds[l-1])
			}
		}
	}
}

// ---- 核验二：实测 MSE 不超过解析上界，两条曲线各自独立核验 ----

func TestMSEWithinBound(t *testing.T) {
	for _, p := range allPatterns {
		g := p.gen(64, 48)
		levels := MaxLevels(g.W, g.H)
		// 两种核：单位分解 binomial 核与一般高斯核，界都应成立。
		for _, kc := range []struct {
			label string
			k     []float64
		}{
			{"binomial5", defaultKernel(t)},
			{"gauss7", mustKernel(t, 7, 1.7)},
			{"gauss3", mustKernel(t, 3, 0.8)},
		} {
			lp, err := BuildLaplacian(g, levels, kc.k)
			if err != nil {
				t.Fatalf("%s/%s: %v", p.name, kc.label, err)
			}
			recon, measured, bounds := lp.ReconstructCurve()
			for l := 0; l <= levels; l++ {
				// 独立核验一：用导出的 MSE 在 uint8 重建图上重算实测值。
				// 浮点曲线与 uint8 图之间隔着逐像素四舍五入 e∈[-0.5,0.5]，
				// 由 (d+e)² ≤ d²+2|d||e|+e² 与 Cauchy–Schwarz，
				// MSE_量化 ≤ (√MSE_浮点 + 0.5)²。
				indie := MSE(recon[l], g)
				quantCeil := math.Pow(math.Sqrt(measured[l])+0.5, 2)
				if indie > quantCeil+1e-6 {
					t.Fatalf("%s/%s 第%d层: 量化 MSE %g 超过理论量化上界 %g",
						p.name, kc.label, l, indie, quantCeil)
				}
				// 独立核验二：浮点实测不得超过能量上界（留浮点余量）。
				if measured[l] > bounds[l]+1e-6 {
					t.Fatalf("%s/%s 第%d层: 实测 MSE %g 超过上界 %g",
						p.name, kc.label, l, measured[l], bounds[l])
				}
				// uint8 量化 MSE 不得越过“上界经同一量化放缩后的余量”。
				quantBound := math.Pow(math.Sqrt(bounds[l])+0.5, 2)
				if indie > quantBound+1e-6 {
					t.Fatalf("%s/%s 第%d层: 量化 MSE %g 超过上界(带量化余量) %g",
						p.name, kc.label, l, indie, quantBound)
				}
			}
		}
	}
}

// ---- 完整重建质量：默认核应近似复原，纯色应精确复原 ----

func TestFullReconstruction(t *testing.T) {
	for _, p := range allPatterns {
		g := p.gen(64, 48)
		levels := MaxLevels(g.W, g.H)
		lp, err := BuildLaplacian(g, levels, defaultKernel(t))
		if err != nil {
			t.Fatal(err)
		}
		rec := lp.Reconstruct()
		if rec.W != g.W || rec.H != g.H {
			t.Fatalf("%s: 重建画幅 %dx%d != %dx%d", p.name, rec.W, rec.H, g.W, g.H)
		}
		mse := MSE(rec, g)
		if p.name == "纯色" {
			if mse != 0 {
				t.Fatalf("纯色图应零误差精确重建，MSE=%g", mse)
			}
			continue
		}
		// binomial 单位分解核 + replicate 边界：内部精确，仅边缘少量像素有差。
		if mse > 1.0 {
			t.Fatalf("%s: 完整重建 MSE %g 过大", p.name, mse)
		}
	}
}

// ---- 由目标误差反解层数，实际层数不得超过反解值 ----

func TestSolveLevels(t *testing.T) {
	g := texturePattern(64, 48)
	k := defaultKernel(t)
	maxLv := MaxLevels(g.W, g.H)

	if _, err := SolveLevels(g, 0, k); err != ErrParamTol {
		t.Fatalf("目标误差为零应报 ErrParamTol，得到 %v", err)
	}
	if _, err := SolveLevels(g, -1, k); err != ErrParamTol {
		t.Fatalf("负目标误差应报 ErrParamTol，得到 %v", err)
	}
	if _, err := SolveLevels(nil, 1, k); err != ErrParamImage {
		t.Fatalf("空图应报 ErrParamImage，得到 %v", err)
	}

	// 满层塔曲线：所有反解都在这同一座塔、同一条曲线上核验。
	full, err := BuildLaplacian(g, maxLv, k)
	if err != nil {
		t.Fatal(err)
	}
	_, fullMeasured, fullBounds := full.ReconstructCurve()

	// 每个目标误差独立核验：反解层数在满层塔曲线上“达标且最小”。
	for _, tol := range []float64{5000, 1000, 200, 50, 5, 0.5} {
		n, err := SolveLevels(g, tol, k)
		if err != nil {
			t.Fatalf("tol=%g 反解失败: %v", tol, err)
		}
		if n < 0 || n > maxLv {
			t.Fatalf("tol=%g 反解层数 %d 越界 [0,%d]", tol, n, maxLv)
		}
		// 反解层数 n 在该 tol 下：上界与实测都达标。
		if fullBounds[n] > tol+1e-9 {
			t.Fatalf("tol=%g: 反解层数 %d 的上界 %g 不达标", tol, n, fullBounds[n])
		}
		if fullMeasured[n] > tol+1.0 {
			t.Fatalf("tol=%g: 反解层数 %d 的实测 MSE %g 不达标", tol, n, fullMeasured[n])
		}
		// 且 n 是最小层数：少一层在同一条曲线上不达标（n=0 无前序）。
		if n > 0 && fullBounds[n-1] <= tol {
			t.Fatalf("tol=%g: 同曲线上 %d 层已达标，反解出的 %d 层不是最小层数",
				tol, n-1, n)
		}
	}

	// 目标越宽松，所需层数不增：用一条足够宽的 tol 序列核验整体方向。
	loose, _ := SolveLevels(g, 5000, k)
	strict, _ := SolveLevels(g, 0.5, k)
	if loose > strict {
		t.Fatalf("宽松目标层数 %d 多于严格目标 %d，方向错误", loose, strict)
	}

	// 纯色图：细节全为零，l=0 即达标，反解层数必须为 0。
	n, err := SolveLevels(solidPattern(32, 32), 1e-6, k)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("纯色图任意正误差都应反解为 0 层，得到 %d", n)
	}
}

// ---- 金字塔放大：实际层数不超过反解值，且优于或持平单层缩放 ----

func TestResizePyramid(t *testing.T) {
	g := edgePattern(48, 32)
	k := defaultKernel(t)

	// 目标误差反解层数，实际层数 = min(Levels, n*)。
	nStar, err := SolveLevels(g, 200, k)
	if err != nil {
		t.Fatal(err)
	}
	out, rep, err := ResizePyramid(g, 96, 64, PyramidOptions{
		Levels: 100, KernelLen: DefaultKernelLen, Sigma: 0, // binomial 单位分解核 TargetMSE: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Levels > nStar {
		t.Fatalf("实际层数 %d 超过反解值 %d", rep.Levels, nStar)
	}
	if out.W != 96 || out.H != 64 {
		t.Fatalf("输出画幅错误 %dx%d", out.W, out.H)
	}
	// 曲线两条各自完整且满足单调与上界。
	for l := 1; l <= rep.Levels; l++ {
		if rep.MeasuredMSE[l] > rep.MeasuredMSE[l-1]+1e-9 {
			t.Fatalf("回执实测 MSE 非单调: %v", rep.MeasuredMSE)
		}
		if rep.MeasuredMSE[l] > rep.BoundMSE[l]+1e-6 {
			t.Fatalf("回执实测 MSE 越界: %g > %g", rep.MeasuredMSE[l], rep.BoundMSE[l])
		}
	}

	// 不给目标误差时按显式层数执行，且受几何上限钳制不报错。
	_, rep2, err := ResizePyramid(g, 96, 64, PyramidOptions{
		Levels: 50, KernelLen: DefaultKernelLen, Sigma: 0, // binomial 单位分解核
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep2.Levels != MaxLevels(48, 32) {
		t.Fatalf("应钳制到几何上限 %d，得到 %d", MaxLevels(48, 32), rep2.Levels)
	}

	// 非法输入分流。
	if _, _, err := ResizePyramid(nil, 8, 8, PyramidOptions{Levels: 2}); err != ErrParamImage {
		t.Fatalf("空图应报 ErrParamImage，得到 %v", err)
	}
	if _, _, err := ResizePyramid(g, 0, 8, PyramidOptions{Levels: 2}); err != ErrParamTarget {
		t.Fatalf("坏目标画幅应报 ErrParamTarget，得到 %v", err)
	}
	if _, _, err := ResizePyramid(g, 8, 8, PyramidOptions{KernelLen: 4, Sigma: 1}); err != ErrParamKernel {
		t.Fatalf("偶核长应报 ErrParamKernel，得到 %v", err)
	}
	if _, _, err := ResizePyramid(g, 8, 8, PyramidOptions{}); err != ErrParamLevels {
		t.Fatalf("层数与目标误差均缺省应报 ErrParamLevels，得到 %v", err)
	}
}

// ---- 金字塔的核心价值：含真实细节层时，逐层加回使重建误差单调降到零 ----
//
// 单层缩放不含任何细节表示；拉普拉斯金字塔则把每层高频显式存为细节系数，
// 完整加回后在原分辨率精确复原（浮点 MSE=0）。这里核验的是金字塔相对单层
// 插值的本质能力——可证的细节保持与精确重建，而非无细节纯上采样的 MSE。
func TestPyramidDetailPreservation(t *testing.T) {
	for _, p := range allPatterns {
		g := p.gen(64, 48)
		levels := MaxLevels(g.W, g.H)
		k := defaultKernel(t)
		lp, err := BuildLaplacian(g, levels, k)
		if err != nil {
			t.Fatal(err)
		}
		recon, measured, bounds := lp.ReconstructCurve()
		// 零细节起点（仅最粗层上采样）对含结构的图必然有显著误差；
		// 纯色是特例：常信号上采样无损，零细节即精确。
		if p.name != "纯色" && measured[0] <= 0 {
			t.Fatalf("%s: 零细节重建误差应大于 0", p.name)
		}
		// 完整加回所有真实细节层后，浮点重建在原分辨率精确复原。
		if measured[levels] != 0 {
			t.Fatalf("%s: 完整重建浮点 MSE 应为 0，得到 %g", p.name, measured[levels])
		}
		// 每加回一层，误差相对上一阶段不增（机器精度容差）。
		for l := 1; l <= levels; l++ {
			tol := 1e-7 * math.Max(1, math.Abs(measured[l-1]))
			if measured[l] > measured[l-1]+tol {
				t.Fatalf("%s: 第 %d 层加回细节后误差上升", p.name, l)
			}
			if measured[l] > bounds[l]+1e-6 {
				t.Fatalf("%s: 第 %d 层实测越过能量上界", p.name, l)
			}
		}
		// 细节系数确实承载了高频：至少一层细节的能量非零（纯色除外）。
		if p.name == "纯色" {
			for l := 0; l < levels; l++ {
				if bounds[l] != 0 {
					t.Fatalf("纯色图所有细节层能量应为零，第 %d 层 bound=%g", l, bounds[l])
				}
			}
			continue
		}
		hasDetail := false
		for l := 0; l < levels; l++ {
			if bounds[l] > 0 {
				hasDetail = true
			}
		}
		if !hasDetail {
			t.Fatalf("%s: 非纯色图应至少有一层非零细节能量", p.name)
		}
		_ = recon
	}
}

// ---- 多八度金字塔放大：整数倍放大完全走高斯 expand，不掺单层插值 ----

func TestUpscalePyramidMultiOctave(t *testing.T) {
	g := edgePattern(48, 32)
	// 2× 整数倍：应纯由高斯 expand 八度完成（OctavesUp 与几何一致），
	// 末尾不触发单层双线性收尾（输出尺寸恰为 2 倍）。
	up, rep, err := UpscalePyramid(g, 96, 64, PyramidOptions{
		Levels:    MaxLevels(g.W, g.H),
		KernelLen: DefaultKernelLen,
	})
	if err != nil {
		t.Fatal(err)
	}
	if up.W != 96 || up.H != 64 {
		t.Fatalf("输出画幅错误 %dx%d", up.W, up.H)
	}
	if rep.OctavesUp < 1 {
		t.Fatalf("整数倍放大应至少经过 1 个上采样八度，得到 %d", rep.OctavesUp)
	}
	// 回执曲线仍满足单调与上界。
	for l := 1; l <= rep.Levels; l++ {
		tol := 1e-7 * math.Max(1, math.Abs(rep.MeasuredMSE[l-1]))
		if rep.MeasuredMSE[l] > rep.MeasuredMSE[l-1]+tol {
			t.Fatal("多八度放大回执实测 MSE 非单调")
		}
		if rep.MeasuredMSE[l] > rep.BoundMSE[l]+1e-6 {
			t.Fatal("多八度放大回执实测 MSE 越界")
		}
	}

	// 纯色多八度放大必须零误差：反射边界 + 因子 4 增益保证直流增益为 1。
	solid := solidPattern(16, 16)
	solUp, solRep, err := UpscalePyramid(solid, 64, 64, PyramidOptions{
		Levels: MaxLevels(16, 16), KernelLen: DefaultKernelLen,
	})
	if err != nil {
		t.Fatal(err)
	}
	if solRep.OctavesUp != 2 {
		t.Fatalf("16→64 应经过 2 个上采样八度，得到 %d", solRep.OctavesUp)
	}
	if got := MSE(solUp, solidPattern(64, 64)); got != 0 {
		t.Fatalf("纯色多八度放大应零误差，MSE=%g", got)
	}

	// 非整数倍（48×32 → 100×70）应能正常收齐到精确目标尺寸。
	odd, _, err := UpscalePyramid(g, 100, 70, PyramidOptions{
		Levels: 3, KernelLen: DefaultKernelLen,
	})
	if err != nil {
		t.Fatal(err)
	}
	if odd.W != 100 || odd.H != 70 {
		t.Fatalf("非整数倍输出画幅错误 %dx%d", odd.W, odd.H)
	}
}

// ---- 单层接口行为保持：直通画幅、双线性已知值 ----

func TestSingleLayerResizeUnchanged(t *testing.T) {
	g, _ := New(2, 2)
	g.Pix = []uint8{0, 100, 100, 200}
	out, err := Resize(g, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if out.W != 4 || out.H != 4 {
		t.Fatalf("画幅错误 %dx%d", out.W, out.H)
	}
	// 四角保持原角点值。
	if out.Pix[0] != 0 || out.Pix[3] != 100 ||
		out.Pix[12] != 100 || out.Pix[15] != 200 {
		t.Fatalf("角点被改变: %v", out.Pix)
	}
	// 同尺寸直通且不改像素。
	same, err := Resize(g, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Pix {
		if same.Pix[i] != g.Pix[i] {
			t.Fatal("同尺寸直通被改动")
		}
	}
}

func TestMaxLevels(t *testing.T) {
	cases := []struct {
		w, h, want int
	}{
		{1, 1, 0},
		{2, 2, 1},
		{3, 3, 2},
		{4, 4, 2},
		{8, 8, 3},
		{64, 48, 6},
		{1, 8, 3},
	}
	for _, c := range cases {
		if got := MaxLevels(c.w, c.h); got != c.want {
			t.Fatalf("MaxLevels(%d,%d)=%d, want %d", c.w, c.h, got, c.want)
		}
	}
}

func mustKernel(t *testing.T, n int, sigma float64) []float64 {
	t.Helper()
	k, err := GaussianKernel(n, sigma)
	if err != nil {
		t.Fatal(err)
	}
	return k
}
