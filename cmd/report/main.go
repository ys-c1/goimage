// 报告生成器：构造四类十六位素材，跑三档桶数扫描与截断对照，
// 输出 artifacts/ 下的 Markdown 报告与 PNG 对照图。仅依赖标准库。
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"time"

	"goimage/img"
)

const (
	outDir    = "artifacts"
	imgW      = 256
	imgH      = 192
	bucketLo  = 16
	bucketMid = 64
	bucketHi  = 256
)

type sample struct {
	key  string
	name string
	g    *img.Gray16
}

func buildSamples() []sample {
	return []sample{
		{"gradient", "渐变", gradient(imgW, imgH)},
		{"photo", "照片", photo(imgW, imgH)},
		{"icon", "图标", icon(imgW, imgH)},
		{"solid", "纯色", solid(imgW, imgH, 32768)},
	}
}

func gradient(w, h int) *img.Gray16 {
	// 浮点插值的非对齐渐变：步长 65535/(w-1)≈257.98，不落在八位阶边界上，
	// 直接截断会把连续过渡压成可见台阶（色带）。
	g, _ := img.NewGray16(w, h)
	for i := range g.Pix {
		x := i % w
		g.Pix[i] = uint16(math.Round(float64(x) * 65535.0 / float64(w-1)))
	}
	return g
}

func photo(w, h int) *img.Gray16 {
	g, _ := img.NewGray16(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			fx, fy := float64(x)/float64(w), float64(y)/float64(h)
			v := 0.5 + 0.25*math.Sin(fx*18) + 0.15*math.Cos(fy*23) + 0.1*math.Sin((fx+fy)*31)
			if v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			g.Pix[y*w+x] = uint16(v * 65535)
		}
	}
	return g
}

func icon(w, h int) *img.Gray16 {
	g, _ := img.NewGray16(w, h)
	cx, cy, r := w/2, h/2, w/3
	tones := []uint16{0, 12000, 34000, 55000, 65535}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			d := (x-cx)*(x-cx) + (y-cy)*(y-cy)
			switch {
			case d < r*r/4:
				g.Pix[y*w+x] = tones[4]
			case d < r*r:
				g.Pix[y*w+x] = tones[2]
			case x == cx || y == cy:
				g.Pix[y*w+x] = tones[3]
			default:
				g.Pix[y*w+x] = tones[1]
			}
		}
	}
	return g
}

func solid(w, h int, tone uint16) *img.Gray16 {
	g, _ := img.NewGray16(w, h)
	for i := range g.Pix {
		g.Pix[i] = tone
	}
	return g
}

type rowResult struct {
	buckets    int
	palSize    int
	maxErr8    float64
	bound8     float64
	meanAbs8   float64
	rms8       float64
	elapsed    time.Duration
	truncMean8 float64
	truncMax8  float64
}

func errStats(g *img.Gray16, q *img.Gray) (mean8, max8 float64) {
	sum, mx := 0.0, 0
	for i, v := range g.Pix {
		ref := uint16(q.Pix[i]) * 257
		d := int(v) - int(ref)
		if d < 0 {
			d = -d
		}
		sum += float64(d)
		if d > mx {
			mx = d
		}
	}
	n := float64(len(g.Pix))
	return sum / n / 257.0, float64(mx) / 257.0
}

// regionMAE / regionMax 只统计前 n 个像素（密集区）的色差，八位阶单位。
func regionMAE(g *img.Gray16, q *img.Gray, n int) float64 {
	sum := 0.0
	for i := 0; i < n; i++ {
		d := int(g.Pix[i]) - int(uint16(q.Pix[i])*257)
		if d < 0 {
			d = -d
		}
		sum += float64(d)
	}
	return sum / float64(n) / 257.0
}

func regionMax(g *img.Gray16, q *img.Gray, n int) float64 {
	mx := 0
	for i := 0; i < n; i++ {
		d := int(g.Pix[i]) - int(uint16(q.Pix[i])*257)
		if d < 0 {
			d = -d
		}
		if d > mx {
			mx = d
		}
	}
	return float64(mx) / 257.0
}

func runCount(p []uint8) int {
	r := 1
	for i := 1; i < len(p); i++ {
		if p[i] != p[i-1] {
			r++
		}
	}
	return r
}

func scan(g *img.Gray16, k int) rowResult {
	start := time.Now()
	_, st, err := img.MedianCut16(g, k)
	if err != nil {
		panic(err)
	}
	elapsed := time.Since(start)
	tr := img.Truncate16To8(g)
	tm, tx := errStats(g, tr)
	return rowResult{
		buckets:    st.Buckets,
		palSize:    len(st.Palette),
		maxErr8:    st.MaxErr8(),
		bound8:     st.Bound8(),
		meanAbs8:   st.MeanAbs8(),
		rms8:       st.RMS8(),
		elapsed:    elapsed,
		truncMean8: tm,
		truncMax8:  tx,
	}
}

// renderStrip 把一张八位灰度图画进 dst 的指定行带。
func renderStrip(dst *image.Gray, q *img.Gray, top int) {
	for y := 0; y < q.H; y++ {
		for x := 0; x < q.W; x++ {
			dst.SetGray(x, top+y, color.Gray{Y: q.Pix[y*q.W+x]})
		}
	}
}

func renderSource16(dst *image.Gray, g *img.Gray16, top int) {
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			dst.SetGray(x, top+y, color.Gray{Y: uint8(g.Pix[y*g.W+x] >> 8)})
		}
	}
}

// writeComparisonPNG 生成五行对照图：源 / 截断 / 16桶 / 64桶 / 256桶。
func writeComparisonPNG(s sample, results map[int]rowResult) error {
	const strips = 5
	gap := 4
	canvas := image.NewGray(image.Rect(0, 0, imgW, strips*imgH+(strips-1)*gap))
	renderSource16(canvas, s.g, 0)
	tr := img.Truncate16To8(s.g)
	renderStrip(canvas, tr, imgH+gap)
	for i, k := range []int{bucketLo, bucketMid, bucketHi} {
		q, _, err := img.MedianCut16(s.g, k)
		if err != nil {
			return err
		}
		renderStrip(canvas, q, (i+2)*(imgH+gap))
	}
	f, err := os.Create(filepath.Join(outDir, "compare_"+s.key+".png"))
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, canvas)
}

// writeBandingPNG 用强偏态渐变展示“同桶数预算”下的色带差异。
// 取单行纵向放大，自上而下四行：
//
//	源（高八位预览）/ 全阶截断 / 等间距 16 阶截断 / 中位切分 16 桶
func writeBandingPNG() error {
	const scale = 64
	w := 512
	g := skewedRamp(w)
	canvas := image.NewGray(image.Rect(0, 0, w, scale*4))
	for y := 0; y < scale; y++ {
		for x := 0; x < w; x++ {
			canvas.SetGray(x, y, color.Gray{Y: uint8(g.Pix[x] >> 8)})
		}
	}
	rows := []*img.Gray{
		img.Truncate16To8(g),
		img.UniformTruncate16(g, 16),
	}
	q16, _, err := img.MedianCut16(g, 16)
	if err != nil {
		return err
	}
	rows = append(rows, q16)
	for band, src := range rows {
		top := (band + 1) * scale
		for y := 0; y < scale; y++ {
			for x := 0; x < w; x++ {
				canvas.SetGray(x, top+y, color.Gray{Y: src.Pix[x]})
			}
		}
	}
	// 行分隔线（深灰），让四行边界清晰。
	for band := 1; band < 4; band++ {
		y := band*scale - 1
		for x := 0; x < w; x++ {
			canvas.SetGray(x, y, color.Gray{Y: 40})
		}
	}
	f, err := os.Create(filepath.Join(outDir, "banding_gradient.png"))
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, canvas)
}

// skewedRamp 构造强偏态单行渐变：前 80% 像素挤在 0..8000 的窄区（仅占
// 全幅约 12%），后 20% 拉开到 65535。等间距 16 阶在密集区只能分到约 2 个阶
// （长色带），中位切分 16 桶则把多数桶投进密集区（短色带、过渡更连续）。
// 注：优势成立于强偏态；分布接近均匀时两者打平，这是中位切分按像素数
// 均衡切分的固有特性，报告中如实说明。
func skewedRamp(w int) *img.Gray16 {
	g, _ := img.NewGray16(w, 1)
	dense := w * 8 / 10
	for x := 0; x < w; x++ {
		var v float64
		if x < dense {
			v = float64(x) / float64(dense) * 8000
		} else {
			t := float64(x-dense) / float64(w-dense)
			v = 8000 + t*(65535-8000)
		}
		g.Pix[x] = uint16(math.Round(v))
	}
	return g
}

func main() {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		panic(err)
	}
	samples := buildSamples()
	buckets := []int{bucketLo, bucketMid, bucketHi}

	type cell struct {
		s sample
		r rowResult
	}
	all := map[string]map[int]rowResult{}
	for _, s := range samples {
		all[s.key] = map[int]rowResult{}
		for _, k := range buckets {
			r := scan(s.g, k)
			all[s.key][k] = r
			if err := writeComparisonPNG(s, all[s.key]); err != nil {
				panic(err)
			}
		}
		writeBandingPNG()
	}

	rep := filepath.Join(outDir, "report.md")
	f, err := os.Create(rep)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := func(format string, a ...interface{}) { fmt.Fprintf(f, format+"\n", a...) }

	w("# 十六位素材出八位版：中位切分调色板量化报告")
	w("")
	w("- 图幅：%d×%d（%d 像素）；桶数扫描三档：%d / %d / %d", imgW, imgH, imgW*imgH, bucketLo, bucketMid, bucketHi)
	w("- 色差统一在十六位空间度量后折算为八位单位（除以 257），数值单位为“八位灰度阶”")
	w("- 耗时为单次量化（建桶 + 映射）墙钟时间，本机测量，见各表末列")
	w("")
	w("## 1. 方差切分依据")
	w("")
	w("中位切分每一轮在当前所有桶中挑选**通道方差最大**的桶，并沿该桶方差最大的通道，")
	w("在中位数处切成两半。依据有三条：")
	w("")
	w("1. **方差即离散度**：方差最大的通道/桶意味着像素在该方向铺得最开、量化后可能贡献的误差最大，")
	w("   优先切它能最快压低整体误差上界；")
	w("2. **中位数保证均衡**：从中位数切分使两侧像素数大致相等，避免出现超大桶长期得不到细分；")
	w("3. **重复值退化处理**：若切点两侧取值相同（离散色/纯色场景），切点挪到第一个异值边界；")
	w("   若整桶同值则标记不可切分并终止——这也是纯色/单像素素材桶数会低于上限的原因。")
	w("")
	w("本实现按通用 N 通道写法实现（`medianCut(vals, channels, maxBuckets)`），灰度图 channels=1，")
	w("方差选轴恒选中唯一通道；同一套代码可直接用于 RGB 三通道，那时方差选轴才会在 R/G/B 间切换。")
	w("调色板项取桶内均值四舍五入到八位，映射阶段对升序调色板做二分最近邻。")
	w("")
	w("## 2. 四类图色差表（按图分栏，三档桶数）")
	for _, s := range samples {
		w("")
		w("### %s（%s）", s.name, s.key)
		w("")
		w("| 桶数上限 | 实际桶数 | 调色板项 | 最大色差 | 色差上界 | 平均绝对色差 | 均方根色差 | 截断(MAE) | 截断(最大) | 耗时 |")
		w("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
		for _, k := range buckets {
			r := all[s.key][k]
			w("| %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s |",
				k, r.buckets, r.palSize, r.maxErr8, r.bound8, r.meanAbs8, r.rms8,
				r.truncMean8, r.truncMax8, r.elapsed.Round(time.Microsecond))
		}
	}
	w("")
	w("> 说明：最大色差与色差上界在本实现中相等——最近邻映射在固定调色板上对每个像素都取偏差最小的项，")
	w("> 任意其它映射的像素偏差都不小于它，因此最近邻的最大偏差就是该调色板下可达的最小最大偏差，是一个可证明的紧上界。")
	w("")
	w("## 3. 桶数扫描：三档横向对比")
	w("")
	w("| 素材 | 指标 | 16 桶 | 64 桶 | 256 桶 |")
	w("|---|---|---:|---:|---:|")
	for _, s := range samples {
		w("| %s | 最大色差 | %.3f | %.3f | %.3f |", s.name,
			all[s.key][16].maxErr8, all[s.key][64].maxErr8, all[s.key][256].maxErr8)
		w("| %s | 平均绝对色差 | %.3f | %.3f | %.3f |", s.name,
			all[s.key][16].meanAbs8, all[s.key][64].meanAbs8, all[s.key][256].meanAbs8)
		w("| %s | 实际桶数 | %d | %d | %d |", s.name,
			all[s.key][16].buckets, all[s.key][64].buckets, all[s.key][256].buckets)
	}
	w("")
	w("## 4. 两档图幅耗时（go test benchmark，另见 `go test -bench=.`）")
	w("")
	w("报告脚本图幅 %d×%d；库内 benchmark 另测 256×256 与 512×512 两档，结果以测试运行为准。", imgW, imgH)
	w("")
	w("## 5. 对照图")
	w("")
	w("每张 `compare_<素材>.png` 自上而下五行，行序固定：")
	w("")
	w("1. 十六位原图（按高八位预览）")
	w("2. 直接截断（右移八位，无调色板）")
	w("3. 中位切分 16 桶")
	w("4. 中位切分 64 桶")
	w("5. 中位切分 256 桶")
	w("")
	for _, s := range samples {
		w("- `compare_%s.png` —— %s 五行对照", s.key, s.name)
	}
	w("- `banding_gradient.png` —— 强偏态渐变单行纵向放大，自上而下四行：")
	w("  源 / 全阶截断（256 阶预算）/ 等间距 16 阶截断 / 中位切分 16 桶。")
	w("  后两行是**同 16 桶预算**的公平对照：强偏态下等间距截断在左侧密集区")
	w("  只分到约 2 个阶、呈现长色带，中位切分把多数桶投进密集区、台阶更短。")
	w("")
	// 偏态渐变的同预算量化对照（与色带图同一数据）。
	skew := skewedRamp(512)
	dense := 512 * 8 / 10
	qMC, _, _ := img.MedianCut16(skew, 16)
	uni := img.UniformTruncate16(skew, 16)
	full := img.Truncate16To8(skew)
	w("### 5.1 偏态渐变同预算对照（密集区 = 前 80%% 像素，单位：八位阶）")
	w("")
	w("| 方案 | 桶数预算 | 密集区 MAE | 密集区最大色差 | 全图平台数 |")
	w("|---|---:|---:|---:|---:|")
	w("| 全阶截断 | 256 | %.3f | %.3f | %d |", regionMAE(skew, full, dense), regionMax(skew, full, dense), runCount(full.Pix))
	w("| 等间距截断 | 16 | %.3f | %.3f | %d |", regionMAE(skew, uni, dense), regionMax(skew, uni, dense), runCount(uni.Pix))
	w("| 中位切分 | 16 | %.3f | %.3f | %d |", regionMAE(skew, qMC, dense), regionMax(skew, qMC, dense), runCount(qMC.Pix))
	w("")
	w("> 适用边界：中位切分按**像素数**均衡切分，优势在分布不均（簇状/偏态）时最明显；")
	w("> 对接近均匀分布的素材，它与等间距截断大致打平。这是该算法的固有特性，而非缺陷。")
	w("> 全阶截断拥有 256 阶预算，密集区误差天然更小，但它不解决“低桶数预算下桶如何分配”的问题。")
	w("")
	w("## 6. 错误码约定")
	w("")
	w("| 触发条件 | 返回错误 |")
	w("|---|---|")
	w("| 桶数上限为 0、负数或 >256 | `ErrBuckets` |")
	w("| 八位色阶非法（<1 或 >256，既有 Quantize 路径） | `ErrLevels` |")
	w("| 空图 / nil / 零尺寸导出 | `ErrEmpty` |")
	w("| PGM 头非法（含八位 PGM 送入十六位加载器） | `ErrHeader` |")
	w("")
	w("## 7. 手算对照")
	w("")
	w("`img/reference.go` 是与生产实现不共享任何代码的朴素参考：插入排序、逐次方差选桶、")
	w("暴力最近邻。测试 `TestMedianCutMatchesHandReference` 对五类素材逐像素比对两套实现，")
	w("`TestHandComputedTinyCase` 锁定四像素两桶的纸面可复算样例（调色板 {20, 228}）。")

	fmt.Println("报告已写入", rep)
}
