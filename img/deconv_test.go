package img

import (
	"errors"
	"math"
	"testing"
)

func testKernel(t *testing.T) Kernel {
	t.Helper()
	k, err := GaussianKernel(3, 0.8)
	if err != nil {
		t.Fatalf("构造核失败：%v", err)
	}
	return k
}

func deconvAll(t *testing.T) ([]DeconvCase, Kernel) {
	t.Helper()
	k := testKernel(t)
	cases, err := DeconvCases(k)
	if err != nil {
		t.Fatalf("构造用例失败：%v", err)
	}
	if len(cases) != len(DeconvCaseNames) {
		t.Fatalf("用例数 %d，期望 %d", len(cases), len(DeconvCaseNames))
	}
	return cases, k
}

// 核验一：振铃幅度随迭代单调不增，且最终不超过半级。
// 与核验二分开放置，互不掺入。
func TestDeconvRingingDecreases(t *testing.T) {
	cases, k := deconvAll(t)
	for _, c := range cases {
		out, st, err := Deconvolve(c.Observed, DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 64,
		})
		if err != nil {
			t.Fatalf("%s：反卷积失败 %v", c.Name, err)
		}
		if len(st.Steps) == 0 {
			t.Fatalf("%s：迭代次数为零", c.Name)
		}
		prev := st.RingStart
		for _, s := range st.Steps {
			if s.RingAmp > prev+1e-9 {
				t.Fatalf("%s：第 %d 轮振铃幅度 %.4f 高于上一轮 %.4f",
					c.Name, s.Iter, s.RingAmp, prev)
			}
			prev = s.RingAmp
		}
		if st.RingEnd > ringLimit {
			t.Fatalf("%s：最终振铃幅度 %.4f 超过半级", c.Name, st.RingEnd)
		}
		amp, _ := RingStats(out, c.Observed)
		if amp > ringLimit {
			t.Fatalf("%s：8 位输出的振铃幅度 %.4f 超过半级", c.Name, amp)
		}
		if !st.Converged || st.Stop != "joint-bound" {
			t.Fatalf("%s：未由联合下界停机（%s）", c.Name, st.Stop)
		}
	}
}

// 残差单项判据会被白边骗过：初值（单次锐化）的残差已在下界之内，
// 但振铃幅度远超半级。故残差必须与振铃共同判定。
func TestDeconvResidualCriterionAloneAcceptsHalo(t *testing.T) {
	cases, k := deconvAll(t)
	for _, c := range cases {
		if c.Name == "纯色" {
			continue
		}
		_, st, err := Deconvolve(c.Observed, DeconvOptions{Kernel: k, Amount: 1, MaxIter: 1})
		if err != nil {
			t.Fatalf("%s：%v", c.Name, err)
		}
		if st.ResidualStart > st.ResidualBound {
			t.Fatalf("%s：初值残差 %.6f 未进入下界 %.6f，前提不成立",
				c.Name, st.ResidualStart, st.ResidualBound)
		}
		if st.RingStart <= ringLimit {
			t.Fatalf("%s：初值振铃 %.4f 未超半级，前提不成立", c.Name, st.RingStart)
		}
	}
}

// 核验二：最终过冲像素数为零。单独统计，不复用核验一的判据。
func TestDeconvFinalOvershootZero(t *testing.T) {
	cases, k := deconvAll(t)
	for _, c := range cases {
		out, st, err := Deconvolve(c.Observed, DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 64,
		})
		if err != nil {
			t.Fatalf("%s：反卷积失败 %v", c.Name, err)
		}
		if st.RingPixelsEnd != 0 {
			t.Fatalf("%s：最终振铃像素 %d，应为零", c.Name, st.RingPixelsEnd)
		}
		// 停机那轮以预算零投影，终值应严格落在局部值域内，而非“不超过半级”。
		if st.RingEnd != 0 {
			t.Fatalf("%s：最终振铃幅度 %.6f，应为零", c.Name, st.RingEnd)
		}
		_, pixels := RingStats(out, c.Observed)
		if pixels != 0 {
			t.Fatalf("%s：8 位输出过冲像素 %d，应为零", c.Name, pixels)
		}
		// 过冲判据在源图局部值域上同样为零：结果不越过干净图的邻域范围。
		_, cleanPixels := RingStats(out, c.Clean)
		if cleanPixels != 0 {
			t.Fatalf("%s：相对源图的过冲像素 %d，应为零", c.Name, cleanPixels)
		}
	}
}

// 对照：单次锐化在白边处大量过冲，反卷积把过冲清零，锐度不降。
func TestDeconvVsUnsharpOvershoot(t *testing.T) {
	cases, k := deconvAll(t)
	for _, c := range cases {
		single, err := SharpenUnsharp(c.Observed, k, 1)
		if err != nil {
			t.Fatalf("%s：单次锐化失败 %v", c.Name, err)
		}
		_, singlePixels := RingStats(single, c.Observed)
		out, _, err := Deconvolve(c.Observed, DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 64,
		})
		if err != nil {
			t.Fatalf("%s：反卷积失败 %v", c.Name, err)
		}
		_, pixels := RingStats(out, c.Observed)
		if pixels != 0 {
			t.Fatalf("%s：反卷积过冲 %d，应为零", c.Name, pixels)
		}
		// 纯色没有边缘，两路都应零过冲，只核验锐度不降。
		if singlePixels > 0 && MaxGradient(out) < MaxGradient(single) {
			t.Fatalf("%s：反卷积锐度 %d 低于单次锐化 %d",
				c.Name, MaxGradient(out), MaxGradient(single))
		}
	}
}

// 停止准则由残差与振铃的共同下界给出：两个条件同时满足才停。
func TestDeconvJointStop(t *testing.T) {
	cases, k := deconvAll(t)
	for _, c := range cases {
		got, st, err := Deconvolve(c.Observed, DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 64,
		})
		if err != nil {
			t.Fatalf("%s：%v", c.Name, err)
		}
		if st.Stop != "joint-bound" || !st.Converged {
			t.Fatalf("%s：停止原因 %q，未走联合下界", c.Name, st.Stop)
		}
		last := st.Steps[len(st.Steps)-1]
		if last.RingAmp > ringLimit {
			t.Fatalf("%s：停机时振铃 %.4f 高于半级", c.Name, last.RingAmp)
		}
		if last.Residual > st.ResidualBound+1e-12 {
			t.Fatalf("%s：停机时残差 %.6f 高于下界 %.6f",
				c.Name, last.Residual, st.ResidualBound)
		}
		// 对外报告的终值必须同样满足两条下界——停机轮的空预算投影会
		// 改动残差，判定须针对投影后的状态。
		if st.ResidualEnd > st.ResidualBound {
			t.Fatalf("%s：最终残差 %.6f 高于下界 %.6f",
				c.Name, st.ResidualEnd, st.ResidualBound)
		}
		if st.RingEnd > ringLimit {
			t.Fatalf("%s：最终振铃 %.4f 超过半级", c.Name, st.RingEnd)
		}
		// 用独立的残差函数复核统计值，避免报表口径与判据同源而互证。
		// 统计值取浮点末轮，8 位输出会引入量化误差，故只要求量级一致。
		res, err := RelResidual(got, c.Observed, k)
		if err != nil {
			t.Fatalf("%s：%v", c.Name, err)
		}
		if math.Abs(res-last.Residual) > 2e-3 {
			t.Fatalf("%s：残差复核 %.6f 与统计 %.6f 不一致", c.Name, res, last.Residual)
		}
		if res > st.ResidualBound {
			t.Fatalf("%s：8 位输出的残差 %.6f 高于下界 %.6f",
				c.Name, res, st.ResidualBound)
		}
		// 提前一轮停时必有一条不满足，说明两个下界都参与。
		if len(st.Steps) > 1 {
			prev := st.Steps[len(st.Steps)-2]
			if prev.RingAmp <= ringLimit && prev.Residual <= st.ResidualBound+1e-12 {
				t.Fatalf("%s：上一轮已满足两条下界，停机过晚", c.Name)
			}
		}
	}
}

// 只看残差会怎样：单次锐化初值的残差已经低于下界，但白边仍在；
// 关掉振铃预算后残差继续下降，振铃却始终停在半级之上。
// 这条对照说明停止准则为何必须同时约束残差与振铃。
func TestDeconvResidualAloneIsInsufficient(t *testing.T) {
	cases, k := deconvAll(t)
	checked := 0
	for _, c := range cases {
		if c.Name == "纯色" {
			continue // 纯色无振铃，不构成对照
		}
		_, st, err := Deconvolve(c.Observed, DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 8, DisableRingBudget: true,
		})
		if err != nil {
			t.Fatalf("%s：%v", c.Name, err)
		}
		first := st.Steps[0]
		last := st.Steps[len(st.Steps)-1]
		if last.Residual >= first.Residual {
			t.Fatalf("%s：残差未下降（%.6f→%.6f），对照失效",
				c.Name, first.Residual, last.Residual)
		}
		// 残差一路下降到下界以下，振铃却始终停在半级之上：
		// 只按残差停机就会把带白边的图当成合格结果收下。
		if last.Residual > st.ResidualBound {
			t.Fatalf("%s：残差 %.6f 未进入下界 %.6f",
				c.Name, last.Residual, st.ResidualBound)
		}
		for _, s := range st.Steps {
			if s.RingAmp <= ringLimit {
				t.Fatalf("%s：无预算时第 %d 轮振铃已降到 %.4f，对照失效",
					c.Name, s.Iter, s.RingAmp)
			}
		}
		checked++
	}
	if checked != 4 {
		t.Fatalf("对照覆盖 %d 幅，期望 4 幅", checked)
	}
}

// 非负与总和约束在每轮均成立。
func TestDeconvConstraints(t *testing.T) {
	cases, k := deconvAll(t)
	for _, c := range cases {
		out, _, err := Deconvolve(c.Observed, DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 64,
		})
		if err != nil {
			t.Fatalf("%s：%v", c.Name, err)
		}
		// 输出是 8 位图，MinPix 恒非负，这条只防导出路径漏钳；
		// 真正检验非负投影的是下面的浮点投影用例。
		if MinPix(out) < 0 {
			t.Fatalf("%s：出现负像素", c.Name)
		}
		// 总和约束在浮点迭代里成立；量化后允许每像素半级的舍入误差。
		diff := math.Abs(Sum(out) - Sum(c.Observed))
		limit := 0.5*float64(len(out.Pix)) + 1
		if diff > limit {
			t.Fatalf("%s：总和偏差 %.1f 超过量化上限 %.1f", c.Name, diff, limit)
		}
	}
}

func TestDeconvRejectsBadInputs(t *testing.T) {
	k := testKernel(t)
	good, _ := New(8, 8)
	// 方框核的频域符号有零点，NewKernel 在构造阶段就会拒绝；
	// 这里要测“使用阶段”的可逆性校验，故直接用结构体字面量绕过构造。
	singular := Kernel{Size: 3, W: []float64{
		1.0 / 9, 1.0 / 9, 1.0 / 9,
		1.0 / 9, 1.0 / 9, 1.0 / 9,
		1.0 / 9, 1.0 / 9, 1.0 / 9,
	}}

	var short Gray = Gray{W: 8, H: 8, Pix: make([]uint8, 10)}
	cases := []struct {
		name  string
		img   *Gray
		opt   DeconvOptions
		field string
		sent  error
	}{
		{"nil 图", nil, DeconvOptions{Kernel: k, MaxIter: 4}, "image", ErrEmptyImage},
		{"宽为零", &Gray{W: 0, H: 8}, DeconvOptions{Kernel: k, MaxIter: 4}, "image", ErrEmptyImage},
		{"像素不足", &short, DeconvOptions{Kernel: k, MaxIter: 4}, "image", ErrEmptyImage},
		{"迭代上限为零", good, DeconvOptions{Kernel: k, MaxIter: 0}, "maxIter", ErrIterations},
		{"迭代上限为负", good, DeconvOptions{Kernel: k, MaxIter: -3}, "maxIter", ErrIterations},
		{"锐化量为负", good, DeconvOptions{Kernel: k, Amount: -1, MaxIter: 4}, "amount", ErrAmount},
		{"锐化量为 NaN", good, DeconvOptions{Kernel: k, Amount: math.NaN(), MaxIter: 4}, "amount", ErrAmount},
		{"核不可逆", good, DeconvOptions{Kernel: singular, MaxIter: 4}, "kernel", ErrKernelSingular},
		{"核为偶数长", good, DeconvOptions{Kernel: Kernel{Size: 2, W: []float64{.25, .25, .25, .25}}, MaxIter: 4}, "kernel.size", ErrKernel},
		{"核元素个数不符", good, DeconvOptions{Kernel: Kernel{Size: 3, W: []float64{1}}, MaxIter: 4}, "kernel", ErrKernel},
		{"核含负权重", good, DeconvOptions{Kernel: Kernel{Size: 3, W: []float64{
			1, 1, 1, 1, -1, 1, 1, 1, 1,
		}}, MaxIter: 4}, "kernel[4]", ErrKernel},
		{"核含 NaN", good, DeconvOptions{Kernel: Kernel{Size: 3, W: []float64{
			1, 1, 1, 1, math.NaN(), 1, 1, 1, 1,
		}}, MaxIter: 4}, "kernel[4]", ErrKernel},
		{"核和为零", good, DeconvOptions{Kernel: Kernel{Size: 3, W: make([]float64, 9)}, MaxIter: 4}, "kernel", ErrKernel},
	}
	for _, c := range cases {
		_, _, err := Deconvolve(c.img, c.opt)
		if err == nil {
			t.Fatalf("%s：应报错", c.name)
		}
		var fe *FieldError
		if !errors.As(err, &fe) {
			t.Fatalf("%s：错误不是 FieldError：%v", c.name, err)
		}
		if fe.FieldName() != c.field {
			t.Fatalf("%s：字段名 %q，期望 %q", c.name, fe.FieldName(), c.field)
		}
		if !errors.Is(err, c.sent) {
			t.Fatalf("%s：错误 %v 未匹配哨兵 %v", c.name, err, c.sent)
		}
	}
}

// 单次锐化的行为保持不变：固定输入逐像素对照冻结公式。
func TestSharpenUnsharpUnchanged(t *testing.T) {
	k := testKernel(t)
	g, _ := New(5, 5)
	for i := range g.Pix {
		g.Pix[i] = uint8((i * 37) % 256)
	}
	got, err := SharpenUnsharp(g, k, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{
		0, 7, 76, 128, 174,
		237, 255, 0, 0, 71,
		105, 166, 235, 255, 0,
		0, 65, 117, 169, 246,
		255, 0, 29, 66, 112,
	}
	for i := range want {
		if got.Pix[i] != want[i] {
			t.Fatalf("第 %d 像素 %d，期望 %d（单次锐化被改动）", i, got.Pix[i], want[i])
		}
	}
	// 再加入口校验：单次锐化不接受负锐化量。
	if _, err := SharpenUnsharp(g, k, -1); err == nil {
		t.Fatal("负锐化量应报错")
	}
	if _, err := SharpenUnsharp(nil, k, 1); err == nil {
		t.Fatal("空图应报错")
	}
}

// 迭代依赖 Kᵀ 的精确性。<Kx,y> 必须等于 <x,Kᵀy>，边界也不能例外——
// 重复边界下正变换是多对一的，转置必须反向散射而非镜像取样。
// 用非对称核与贴近边界的小画幅才能暴露镜像近似的错误。
func TestKernelAdjoint(t *testing.T) {
	kernels := []struct {
		name string
		w    []float64
	}{
		{"非对称 3x3", []float64{0.1, 0.25, 0.05, 0.15, 0.2, 0.02, 0.03, 0.1, 0.1}},
		{"高斯 3x3", func() []float64 { k, _ := GaussianKernel(3, 0.8); return k.W }()},
		{"非对称 5x5", func() []float64 {
			w := make([]float64, 25)
			for i := range w {
				w[i] = float64(i + 1)
			}
			return w
		}()},
	}
	for _, kc := range kernels {
		n := int(math.Round(math.Sqrt(float64(len(kc.w)))))
		k, err := normalizeKernel(kc.w, n)
		if err != nil {
			t.Fatalf("%s：%v", kc.name, err)
		}
		for _, dim := range [][2]int{{9, 7}, {1, 5}, {5, 1}, {2, 2}, {12, 12}} {
			w, h := dim[0], dim[1]
			x := make([]float64, w*h)
			y := make([]float64, w*h)
			for i := range x {
				x[i] = float64((i*7)%11) - 3
			}
			for i := range y {
				y[i] = float64((i*5)%13) - 6
			}
			kx := k.apply(x, w, h, false)
			kty := k.apply(y, w, h, true)
			var lhs, rhs float64
			for i := range x {
				lhs += kx[i] * y[i]
				rhs += x[i] * kty[i]
			}
			if math.Abs(lhs-rhs) > 1e-9 {
				t.Fatalf("%s %dx%d：<Kx,y>=%.8f 与 <x,Kᵀy>=%.8f 不等",
					kc.name, w, h, lhs, rhs)
			}
		}
	}
}

// 投影本身的可加性：非负、盒约束与总和同时被满足。
// 直接测浮点投影，不受 8 位量化的掩盖。
func TestProjectSumBox(t *testing.T) {
	y := []float64{-40, -0.25, 10, 120, 300, 255.5}
	L := []float64{0, 0, 5, 0, 0, 0}
	U := []float64{255, 255, 50, 200, 255, 255}
	want := 700.0
	got := projectSumBox(y, L, U, want)
	var sum float64
	for i := range got {
		if got[i] < L[i]-1e-9 || got[i] > U[i]+1e-9 {
			t.Fatalf("第 %d 项 %.6f 越出盒 [%v,%v]", i, got[i], L[i], U[i])
		}
		sum += got[i]
	}
	if math.Abs(sum-want) > 1e-6 {
		t.Fatalf("总和 %.9f，期望 %.1f", sum, want)
	}
	// 不可达的总和（超过盒上界之和）应钳到上界，不产生 NaN。
	over := projectSumBox(y, L, U, 1e6)
	for i := range over {
		if math.IsNaN(over[i]) || math.Abs(over[i]-U[i]) > 1e-9 {
			t.Fatalf("超界总和未钳到上界：第 %d 项 %.6f 期望 %v", i, over[i], U[i])
		}
	}
}

// 迭代上限不足时如实报告 max-iter，不伪报收敛。
func TestDeconvMaxIterReported(t *testing.T) {
	cases, k := deconvAll(t)
	out, st, err := Deconvolve(cases[0].Observed, DeconvOptions{Kernel: k, Amount: 1, MaxIter: 1})
	if err != nil {
		t.Fatal(err)
	}
	if st.Converged || st.Stop != "max-iter" {
		t.Fatalf("单轮上限不应收敛，得到 %q", st.Stop)
	}
	if st.Iterations != 1 || len(st.Steps) != 1 {
		t.Fatalf("迭代次数 %d，期望 1", st.Iterations)
	}
	if out == nil {
		t.Fatal("应返回最后一轮结果")
	}
}

// 核校验：高斯核归一化、方框核在构造阶段即被拒。
func TestKernelValidation(t *testing.T) {
	k, err := GaussianKernel(3, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, v := range k.W {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("核和 %.15f，应为 1", sum)
	}
	if _, err := GaussianKernel(4, 0.8); err == nil {
		t.Fatal("偶数核长应报错")
	}
	if _, err := GaussianKernel(3, 0); err == nil {
		t.Fatal("sigma 为零应报错")
	}
	// 3x3 高斯的一维符号在 σ > 1/√(2·ln2) ≈ 0.8493 时过零，须判为不可逆。
	if _, err := GaussianKernel(3, 0.84); err != nil {
		t.Fatalf("σ=0.84 应接受：%v", err)
	}
	if _, err := GaussianKernel(3, 0.85); !errors.Is(err, ErrKernelSingular) {
		t.Fatalf("σ=0.85 应判为不可逆，得到 %v", err)
	}
	if _, err := GaussianKernel(5, 1.2); !errors.Is(err, ErrKernelSingular) {
		t.Fatalf("5x5 σ=1.2 应判为不可逆，得到 %v", err)
	}
	if _, err := NewKernel(make([]float64, 9), 3); err == nil {
		t.Fatal("全零核应报错")
	}
	// 3×3 方框核的频域符号在 Nyquist 处为零，必须判为不可逆。
	if _, err := NewKernel([]float64{
		1.0 / 9, 1.0 / 9, 1.0 / 9,
		1.0 / 9, 1.0 / 9, 1.0 / 9,
		1.0 / 9, 1.0 / 9, 1.0 / 9,
	}, 3); !errors.Is(err, ErrKernelSingular) {
		t.Fatalf("方框核应判为不可逆，得到 %v", err)
	}
}
