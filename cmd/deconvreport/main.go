// deconvreport 生成反卷积核验报表：五幅用例的迭代次数、残差、振铃统计。
//
//	go run ./cmd/deconvreport            # 写到 artifacts/deconv-report.md
//	go run ./cmd/deconvreport -o out.md  # 指定输出
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"goimage/img"
)

func main() {
	out := flag.String("o", filepath.Join("artifacts", "deconv-report.md"), "输出路径")
	flag.Parse()

	k, err := img.GaussianKernel(3, 0.8)
	if err != nil {
		fail(err)
	}
	cases, err := img.DeconvCases(k)
	if err != nil {
		fail(err)
	}
	var b strings.Builder
	b.WriteString("# 迭代反卷积：振铃与残差核验\n\n")
	b.WriteString("已知模糊核：3×3 高斯，σ=0.8（归一化，频域可逆）。\n")
	b.WriteString("迭代：Landweber（x ← x + Kᵀ(g−Kx)），每轮后施加非负约束、总和约束，\n")
	b.WriteString("以及按上一轮振铃的一半收缩的局部值域预算。\n")
	b.WriteString("振铃判据：输出相对**观测图** 3×3 局部值域的最大越界灰度级；\n")
	b.WriteString("过冲像素：越界超过半级（0.5）的像素数。\n\n")

	b.WriteString("## 1. 汇总\n\n")
	b.WriteString("| 用例 | 迭代 | 停止 | 残差(初值) | 残差下界 | 残差(终) | 振铃幅度(初→终) | 振铃像素(初→终) | 单次锐化过冲像素 | 反卷积过冲像素 | 最大梯度(观测→反卷积) | 耗时 |\n")
	b.WriteString("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

	type result struct {
		name     string
		obs      *img.Gray
		stats    *img.DeconvStats
		out      *img.Gray
		singlePx int
		elapsed  string
	}
	var results []result
	for _, c := range cases {
		outImg, st, err := img.Deconvolve(c.Observed, img.DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 64,
		})
		if err != nil {
			fail(err)
		}
		unsharp, err := img.SharpenUnsharp(c.Observed, k, 1)
		if err != nil {
			fail(err)
		}
		_, singlePx := img.RingStats(unsharp, c.Observed)
		results = append(results, result{
			name: c.Name, obs: c.Observed, stats: st, out: outImg, singlePx: singlePx,
			elapsed: fmt.Sprintf("%.3fms", float64(st.Elapsed.Microseconds())/1000),
		})
	}
	// 同一次运行的结果同时供汇总表与逐轮轨迹，避免两处数字来自不同运行。
	for _, r := range results {
		st := r.stats
		fmt.Fprintf(&b, "| %s | %d | %s | %.5f | %.5f | %.5f | %.3f→%.3f | %d→%d | %d | %d | %d→%d | %s |\n",
			r.name, st.Iterations, st.Stop, st.ResidualStart, st.ResidualBound, st.ResidualEnd,
			st.RingStart, st.RingEnd, st.RingPixelsStart, st.RingPixelsEnd,
			r.singlePx, st.RingPixelsEnd,
			img.MaxGradient(r.obs), img.MaxGradient(r.out), r.elapsed)
	}

	b.WriteString("\n## 2. 逐轮轨迹\n\n")
	for _, r := range results {
		st := r.stats
		fmt.Fprintf(&b, "### %s（%d 次，%s）\n\n", r.name, st.Iterations, st.Stop)
		b.WriteString("| 步 | 残差 | 振铃幅度 | 振铃像素 | 预算 |\n|---|---|---|---|---|\n")
		for _, s := range st.Steps {
			fmt.Fprintf(&b, "| %d | %.5f | %.4f | %d | %.4f |\n",
				s.Iter, s.Residual, s.RingAmp, s.RingPixels, s.Budget)
		}
		b.WriteString("\n")
	}

	b.WriteString("## 3. 判据与对照\n\n")
	mono := true
	zeroPx := true
	for _, r := range results {
		prev := r.stats.RingStart
		for _, s := range r.stats.Steps {
			if s.RingAmp > prev+1e-9 {
				mono = false
			}
			prev = s.RingAmp
		}
		if r.stats.RingEnd > 0.5 {
			mono = false
		}
		if r.stats.RingPixelsEnd != 0 {
			zeroPx = false
		}
	}
	fmt.Fprintf(&b, "- 振铃幅度随迭代单调不增（五幅）：%v\n", mono)
	fmt.Fprintf(&b, "- 最终过冲像素数为零（五幅）：%v\n", zeroPx)

	// 对照：关掉振铃预算，只保留非负与总和约束。
	b.WriteString("\n对照：关闭振铃预算、只保留非负与总和约束时的行为。\n\n")
	b.WriteString("| 用例 | 首轮残差 | 末轮残差 | 残差下界 | 首轮振铃 | 末轮振铃 | 末轮过冲像素 |\n|---|---|---|---|---|---|---|\n")
	for _, c := range cases {
		_, st, err := img.Deconvolve(c.Observed, img.DeconvOptions{
			Kernel: k, Amount: 1, MaxIter: 8, DisableRingBudget: true,
		})
		if err != nil {
			fail(err)
		}
		first, last := st.Steps[0], st.Steps[len(st.Steps)-1]
		fmt.Fprintf(&b, "| %s | %.5f | %.5f | %.5f | %.3f | %.3f | %d |\n",
			c.Name, first.Residual, last.Residual, st.ResidualBound,
			first.RingAmp, last.RingAmp, last.RingPixels)
	}

	b.WriteString("\n## 4. 停止准则及其理由\n\n")
	b.WriteString("停止需同时满足两条下界：\n\n")
	b.WriteString("1. 残差 ≤ 残差下界（平凡解 x=g 的相对 L2 残差）；\n")
	b.WriteString("2. 振铃幅度 ≤ 半级（0.5 灰阶）。停机那轮以预算零再投影一次，\n")
	b.WriteString("   终值严格落在局部值域内（幅度 0），8 位输出过冲为零。\n\n")
	b.WriteString("只看残差不成立：Landweber 是半收敛的——迭代压低数据残差的同时\n")
	b.WriteString("放大被核压掉的频段，白边随之出现。上表对照里残差一路下降、\n")
	b.WriteString("振铃却始终停在半级之上；单次锐化初值的残差甚至已经低于下界，\n")
	b.WriteString("但它正是带白边的那张图。因此残差与振铃必须共同判定。\n")

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		fail(err)
	}
	if err := os.WriteFile(*out, []byte(b.String()), 0o644); err != nil {
		fail(err)
	}
	fmt.Printf("已写入 %s（%d 幅用例）\n", *out, len(results))
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "deconvreport:", err)
	os.Exit(1)
}
