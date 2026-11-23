// 报告生成器：对五组合成图对跑曝光补偿核验，输出 artifacts/ 下的 Markdown 报告。
// 仅依赖标准库。
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"goimage/img"
	"goimage/internal/testimg"
)

type row struct {
	pair testimg.Pair
	fit  *img.ExposureFit
	err  error
}

func collect() []row {
	var rows []row
	for _, p := range testimg.All() {
		fit, err := img.FitExposure(p.A, p.B, p.Overlap)
		rows = append(rows, row{pair: p, fit: fit, err: err})
	}
	return rows
}

func f4(v float64) string { return fmt.Sprintf("%.4f", v) }

func gainBias(fit *img.ExposureFit) (string, string) {
	return fmt.Sprintf("%.6f / %.6f", fit.Gain[0], fit.Gain[1]),
		fmt.Sprintf("%+.4f / %+.4f", fit.Bias[0], fit.Bias[1])
}

func ratio(before, after float64) string {
	switch {
	case before == 0 && after == 0:
		return "∞"
	case after == 0:
		return "∞"
	case before == 0:
		return "0×"
	default:
		return fmt.Sprintf("%.1f×", before/after)
	}
}

func main() {
	out := flag.String("out", "artifacts/exposure-report.md", "报告输出路径")
	flag.Parse()

	rows := collect()

	var b strings.Builder
	fmt.Fprintf(&b, "# 曝光补偿核验报告\n\n")
	fmt.Fprintf(&b, "- 图对尺寸 %d×%d，水平重叠 %d 列；两图重叠区对应同一组场景点\n",
		testimg.PairW, testimg.PairH, testimg.Overlap)
	fmt.Fprintf(&b, "- 模型：非基准图施加 `I' = gain·I + bias`，使重叠区同一场景点亮度相等；基准图增益固定为 1、偏置固定为 0\n")
	fmt.Fprintf(&b, "- 拟合只用非饱和重叠样本（两侧都严格在 (0,255) 内）；0/255 在补偿前已被截断，不满足线性曝光关系\n")
	fmt.Fprintf(&b, "- 残差为重叠区平均绝对亮度差，按最终 8 位输出计算；“可用”剔除饱和样本，“全部”保留作对照\n")
	fmt.Fprintf(&b, "- 两条要求各自独立核验，互不掺入\n\n")

	// ---- 表 1：增益、偏置、残差三列并列 ----
	fmt.Fprintf(&b, "## 1　增益、偏置、残差\n\n")
	fmt.Fprintf(&b, "| 用例 | 基准图 | 增益（图 1 / 图 2） | 偏置（图 1 / 图 2） | 残差 前 → 后（压缩倍数） | 可用/饱和样本 | 回退图 1 | 欠定 |\n")
	fmt.Fprintf(&b, "|---|---:|---|---|---|---:|---:|---|\n")
	for _, r := range rows {
		if r.err != nil {
			fmt.Fprintf(&b, "| %s | — | 拒绝：%v | — | — | — | — | — |\n", r.pair.Name, r.err)
			continue
		}
		g, bi := gainBias(r.fit)
		fmt.Fprintf(&b, "| %s | 图 1 | %s | %s | %s → %s（%s） | %d / %d | %d | %v |\n",
			r.pair.Name, g, bi,
			f4(r.fit.ResidualBefore), f4(r.fit.ResidualAfter),
			ratio(r.fit.ResidualBefore, r.fit.ResidualAfter),
			r.fit.Usable, r.fit.Saturated, r.fit.FallbackToBase, r.fit.Undetermined)
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "「回退图 1」指重叠区中图 2 已落在 0/255、图 1 仍严格在 (0,255) 内的像素数；\n")
	fmt.Fprintf(&b, "落位时这些像素改用图 1，避免最终输出残留饱和块。无饱和时为 0，与直通拼接落位逐位一致。\n\n")
	fmt.Fprintf(&b, "「全部样本残差」对照（饱和像素保留在内），以及按实际落位策略\n")
	fmt.Fprintf(&b, "（饱和像素回退图 1）后重叠区的最终残差：\n\n")
	fmt.Fprintf(&b, "| 用例 | 全部样本 残差前 | 全部样本 残差后 | 落位后 重叠区残差 | 说明 |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---|\n")
	for _, r := range rows {
		if r.err != nil {
			continue
		}
		note := "—"
		if r.fit.Saturated > 0 {
			note = "饱和像素在校正前已丢失信息，无法恢复；落位时回退图 1"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s |\n",
			r.pair.Name, f4(r.fit.ResidualBeforeAll), f4(r.fit.ResidualAfterAll),
			f4(r.fit.ResidualAfterPlaced), note)
	}
	fmt.Fprintf(&b, "\n")

	// ---- 核验一 ----
	fmt.Fprintf(&b, "## 2　核验一：重叠区亮度差至少缩小一个量级\n\n")
	fmt.Fprintf(&b, "| 用例 | 残差前 | 残差后 | 实际压缩 | 阈值 | 判定 |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---|\n")
	for _, r := range rows {
		if r.err != nil {
			continue
		}
		verdict := "通过"
		if !r.fit.OrderReduced() {
			verdict = "不通过"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | 10× | %s |\n",
			r.pair.Name, f4(r.fit.ResidualBefore), f4(r.fit.ResidualAfter),
			ratio(r.fit.ResidualBefore, r.fit.ResidualAfter), verdict)
	}
	fmt.Fprintf(&b, "\n")

	// ---- 核验二 ----
	fmt.Fprintf(&b, "## 3　核验二：不重叠区不出现截断\n\n")
	fmt.Fprintf(&b, "| 用例 | 越界像素数 | 非重叠区校正后范围 | 判定 |\n")
	fmt.Fprintf(&b, "|---|---:|---|---|\n")
	for _, r := range rows {
		if r.err != nil {
			continue
		}
		verdict := "通过"
		if r.fit.NonOverlapClip != 0 {
			verdict = "不通过"
		}
		fmt.Fprintf(&b, "| %s | %d | [%s, %s] | %s |\n",
			r.pair.Name, r.fit.NonOverlapClip,
			f4(r.fit.NonOverlapMin), f4(r.fit.NonOverlapMax), verdict)
	}
	fmt.Fprintf(&b, "\n")

	// ---- 欠定与正则项 ----
	fmt.Fprintf(&b, "## 4　欠定与正则项\n\n")
	for _, r := range rows {
		if r.err != nil || !r.fit.Undetermined {
			continue
		}
		fmt.Fprintf(&b, "- **%s**：重叠区只有单一亮度，Σ(x−x̄)²≈0，增益不可辨识。\n", r.pair.Name)
		fmt.Fprintf(&b, "  实际使用 λ=%.4f，增益由正则项固定为 %.6f，偏置吸收均值差（%+.4f）。\n",
			r.fit.Lambda, r.fit.Gain[1], r.fit.Bias[1])
	}
	fmt.Fprintf(&b, "\n正则项取 Tikhonov 形式 λ·(g−1)²，λ=1e-6·(Σx²+n)：只惩罚增益不惩罚偏置，\n")
	fmt.Fprintf(&b, "因为平坦区里偏移由数据唯一确定而对比度无信息；惩罚中心取 g=1，因为“不改变对比度”\n")
	fmt.Fprintf(&b, "是唯一非任意先验，且不会把量化噪声按任意倍数放大。设重叠区标准差 σ，则\n")
	fmt.Fprintf(&b, "λ/sxx≈1e-6·(1+(x̄²+1)/σ²)，纹理明显时增益偏移不足一个码值，解与普通最小二乘一致。\n\n")

	// ---- 拒绝路径 ----
	fmt.Fprintf(&b, "## 5　拒绝路径（均报出图号）\n\n")
	for _, line := range rejectionLines() {
		fmt.Fprintf(&b, "- %s\n", line)
	}
	fmt.Fprintf(&b, "\n")

	// ---- 直通 ----
	fmt.Fprintf(&b, "## 6　直通拼接保持不变\n\n")
	fmt.Fprintf(&b, "| 用例 | 直通拼接输出画幅 | 与输入逐位一致 |\n")
	fmt.Fprintf(&b, "|---|---|---|\n")
	for _, r := range rows {
		out, err := img.StitchDirect(r.pair.A, r.pair.B, r.pair.Overlap)
		if err != nil {
			fmt.Fprintf(&b, "| %s | — | 拒绝：%v |\n", r.pair.Name, err)
			continue
		}
		fmt.Fprintf(&b, "| %s | %d×%d | %v |\n",
			r.pair.Name, out.W, out.H, directMatchesInput(out, r.pair))
	}
	fmt.Fprintf(&b, "\n")

	// ---- 复现命令（最后一节） ----
	fmt.Fprintf(&b, "## 7　复现命令\n\n")
	fmt.Fprintf(&b, "```sh\n")
	fmt.Fprintf(&b, "# 单元测试：五组用例，两条核验各自独立的断言\n")
	fmt.Fprintf(&b, "go test ./img -run 'TestExposure|TestStitch' -v -count=1\n\n")
	fmt.Fprintf(&b, "# 全量测试（含竞态）\n")
	fmt.Fprintf(&b, "go test ./... -race -count=1\n\n")
	fmt.Fprintf(&b, "# 生成本报告（表 1 的增益/偏置/残差三列并列）\n")
	fmt.Fprintf(&b, "go run ./cmd/exposurereport -out %s\n", *out)
	fmt.Fprintf(&b, "```\n")

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, []byte(b.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(*out)
}

func directMatchesInput(out *img.Gray, p testimg.Pair) bool {
	want, err := img.New(testimg.SceneW, testimg.PairH)
	if err != nil {
		return false
	}
	for y := 0; y < p.A.H; y++ {
		copy(want.Pix[y*want.W:y*want.W+p.A.W], p.A.Pix[y*p.A.W:y*p.A.W+p.A.W])
		for x := p.Overlap; x < p.B.W; x++ {
			want.Pix[y*want.W+p.A.W+(x-p.Overlap)] = p.B.Pix[y*p.B.W+x]
		}
	}
	if out.W != want.W || out.H != want.H {
		return false
	}
	for i := range want.Pix {
		if out.Pix[i] != want.Pix[i] {
			return false
		}
	}
	return true
}

// rejectionLines 实际触发各条拒绝路径，把真实错误信息照抄进报告。
func rejectionLines() []string {
	p := testimg.All()[0]
	var lines []string
	addLabeled := func(label string, err error) {
		if err == nil {
			lines = append(lines, label+"：未拒绝（异常）")
			return
		}
		lines = append(lines, label+"：`"+err.Error()+"`")
	}

	_, err := img.FitExposure(p.A, p.B, 0)
	addLabeled("重叠区为空", err)

	_, err = img.FitExposure(nil, p.B, p.Overlap)
	addLabeled("图为空：图 1", err)

	_, err = img.FitExposure(p.A, nil, p.Overlap)
	addLabeled("图为空：图 2", err)

	_, err = img.FitExposure(p.A, &img.Gray{}, p.Overlap)
	addLabeled("画幅为零：图 2", err)

	_, err = img.FitExposure(p.A, p.B, testimg.PairW+1)
	addLabeled("重叠宽度越界", err)

	short, _ := img.New(testimg.PairW, testimg.PairH-1)
	_, err = img.FitExposure(p.A, short, p.Overlap)
	addLabeled("高度不一致", err)

	negA, _ := img.New(16, 4)
	negB, _ := img.New(16, 4)
	av := []uint8{10, 40, 80, 120}
	bv := []uint8{200, 170, 130, 90}
	for y := 0; y < 4; y++ {
		for k := 0; k < 4; k++ {
			negA.Pix[y*16+(16-4+k)] = av[k]
			negB.Pix[y*16+k] = bv[k]
		}
	}
	_, err = img.FitExposure(negA, negB, 4)
	addLabeled("增益为负", err)

	return lines
}
