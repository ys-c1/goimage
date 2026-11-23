// 报告生成器：在五类合成图案、两档倍率下对比最近邻、双三次与边缘导向放大，
// 输出 artifacts/ 下的 Markdown 报告。仅依赖标准库。
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"goimage/img"
	"goimage/internal/testimg"
)

const outDir = "artifacts"

type method struct {
	key  string
	name string
	fn   func(*img.Gray, float64) (*img.Gray, error)
}

var methods = []method{
	{"nearest", "最近邻", img.UpscaleNearest},
	{"bicubic", "双三次", img.UpscaleBicubic},
	{"edge", "边缘导向", img.UpscaleEdgeDirected},
}

type result struct {
	out *img.Gray
	dur time.Duration
	err error
}

func (r result) ok() bool { return r.err == nil }

func (r result) ssim(ref *img.Gray) float64 { return img.SSIM(r.out, ref) }

type row struct {
	pattern string
	res     map[string]result
}

// measure 取 5 次运行中最快一次作为耗时，返回该次输出。
func measure(g *img.Gray, scale float64, fn func(*img.Gray, float64) (*img.Gray, error)) result {
	var best result
	for i := 0; i < 5; i++ {
		start := time.Now()
		out, err := fn(g, scale)
		d := time.Since(start)
		if err != nil {
			return result{err: err, dur: d}
		}
		if i == 0 || d < best.dur {
			best = result{out: out, dur: d}
		}
	}
	return best
}

func collect() (map[int][]row, map[int][]string) {
	rows := map[int][]row{}
	notes := map[int][]string{}
	for _, sc := range []int{2, 4} {
		scale := float64(sc)
		for _, p := range testimg.All() {
			low := testimg.Downsample(p.Ref, sc)
			r := row{pattern: p.Name, res: map[string]result{}}
			for _, m := range methods {
				res := measure(low, scale, m.fn)
				r.res[m.key] = res
				if !res.ok() {
					notes[sc] = append(notes[sc], fmt.Sprintf("%s 的%s被拒绝（字段 %s）：%v",
						p.Name, m.name, fieldName(res.err), res.err))
				}
			}
			rows[sc] = append(rows[sc], r)
		}
	}
	return rows, notes
}

// fieldName 取下层 FieldError 报告的字段名。
func fieldName(err error) string {
	for err != nil {
		if fe, ok := err.(interface{ FieldName() string }); ok {
			return fe.FieldName()
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			break
		}
		err = u.Unwrap()
	}
	return "?"
}

func fmtDur(d time.Duration) string { return d.Round(time.Microsecond).String() }

func fmtSSIM(res result, ref *img.Gray) string {
	if !res.ok() {
		return "拒绝"
	}
	return fmt.Sprintf("%.4f", res.ssim(ref))
}

func fmtSsimDelta(a, b result, ref *img.Gray) string {
	if !a.ok() || !b.ok() {
		return "—"
	}
	return fmt.Sprintf("%+.4f", a.ssim(ref)-b.ssim(ref))
}

func fmtOver(res result, src *img.Gray, footprint bool) (string, string) {
	if !res.ok() {
		return "—", "—"
	}
	var o img.Overshoot
	if footprint {
		o = img.OvershootStatsFootprint(res.out, src)
	} else {
		o = img.OvershootStats(res.out, src)
	}
	return fmt.Sprintf("%d/%d", o.Pixels, o.Max), fmt.Sprintf("%d", o.Sum)
}

func fmtDiff(a, b *img.Gray) string {
	d := img.DiffStats(a, b)
	return fmt.Sprintf("%.3f | %d", d.MAE, d.Max)
}

func main() {
	data, notes := collect()
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var b []byte
	add := func(format string, args ...any) { b = append(b, fmt.Sprintf(format, args...)...) }

	add("# 边缘导向放大：相似度与过冲核验\n\n")
	add("- 用例：直边、斜边、细线、纹理、纯色；参考图 128×128，输入为盒式降采样后的低分辨率图\n")
	add("- 两档倍率：2×、4×；输出回到 128×128 后与参考图比较\n")
	add("- 相似度为结构相似度 SSIM（高斯窗 11×11，σ=1.5，C1/C2 取标准值）\n")
	add("- 过冲判据：输出像素越出**源图局部 3×3 值域**即计为过冲像素；幅度为越界灰度级\n")
	add("- 末列为更严格的 2×2 双线性足迹判据（与实现限幅所用值域一致），用于交叉核验\n")
	add("- 差值统计单位为八位灰度级；耗时为 5 次取最快一次的墙钟时间，本机测量\n\n")

	add("## 1. 相似度对照（两种基线并列一行）\n\n")
	for _, sc := range []int{2, 4} {
		add("### %d× 倍率\n\n", sc)
		add("| 用例 | 最近邻 | 双三次 | 边缘导向 | 边缘导向−双三次 |\n")
		add("|---|---:|---:|---:|---:|\n")
		for _, r := range data[sc] {
			ref := findRef(r.pattern)
			add("| %s | %s | %s | %s | %s |\n", r.pattern,
				fmtSSIM(r.res["nearest"], ref), fmtSSIM(r.res["bicubic"], ref),
				fmtSSIM(r.res["edge"], ref), fmtSsimDelta(r.res["edge"], r.res["bicubic"], ref))
		}
		add("\n")
	}

	add("## 2. 过冲统计（按档）\n\n")
	for _, sc := range []int{2, 4} {
		add("### %d× 倍率\n\n", sc)
		add("| 用例 | 最近邻 过冲像素/最大幅度 | 双三次 过冲像素/最大幅度 | 边缘导向 过冲像素/最大幅度 | 边缘导向 过冲总量 | 边缘导向 2×2 足迹过冲 |\n")
		add("|---|---:|---:|---:|---:|---:|\n")
		for _, r := range data[sc] {
			low := testimg.Downsample(findRef(r.pattern), sc)
			nOver, _ := fmtOver(r.res["nearest"], low, false)
			bOver, _ := fmtOver(r.res["bicubic"], low, false)
			eOver, eSum := fmtOver(r.res["edge"], low, false)
			eFoot, _ := fmtOver(r.res["edge"], low, true)
			add("| %s | %s | %s | %s | %s | %s |\n", r.pattern, nOver, bOver, eOver, eSum, eFoot)
		}
		add("\n")
	}

	add("## 3. 差值统计（边缘导向 对 双三次、对 参考图）\n\n")
	for _, sc := range []int{2, 4} {
		add("### %d× 倍率\n\n", sc)
		add("| 用例 | 对双三次 MAE | 对双三次 最大差 | 对参考图 MAE | 对参考图 最大差 |\n")
		add("|---|---:|---:|---:|---:|\n")
		for _, r := range data[sc] {
			ref := findRef(r.pattern)
			bic, edge := r.res["bicubic"], r.res["edge"]
			diffBic, diffRef := "— | —", "— | —"
			if edge.ok() && bic.ok() {
				diffBic = fmtDiff(edge.out, bic.out)
			}
			if edge.ok() {
				diffRef = fmtDiff(edge.out, ref)
			}
			add("| %s | %s | %s |\n", r.pattern, diffBic, diffRef)
		}
		add("\n")
	}

	add("## 4. 耗时（每次放大的墙钟时间）\n\n")
	add("| 倍率 | 用例 | 最近邻 | 双三次 | 边缘导向 |\n")
	add("|---:|---|---:|---:|---:|\n")
	for _, sc := range []int{2, 4} {
		for _, r := range data[sc] {
			add("| %d× | %s | %s | %s | %s |\n", sc, r.pattern,
				fmtDur(r.res["nearest"].dur), fmtDur(r.res["bicubic"].dur), fmtDur(r.res["edge"].dur))
		}
	}
	add("\n")

	if len(notes[2])+len(notes[4]) > 0 {
		add("## 注记\n\n")
		for _, sc := range []int{2, 4} {
			for _, n := range notes[sc] {
				add("- %d×：%s\n", sc, n)
			}
		}
		add("\n")
	}

	path := filepath.Join(outDir, "upscale-report.md")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", path)
}

func findRef(name string) *img.Gray {
	for _, p := range testimg.All() {
		if p.Name == name {
			return p.Ref
		}
	}
	panic("未知用例 " + name)
}
