package img

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// Gray16 是高位深灰度图，样值位深由 CompressOptions.SrcDepth 声明。
type Gray16 struct {
	W, H int
	Pix  []uint16
}

func New16(w, h int) (*Gray16, error) {
	if w <= 0 || h <= 0 {
		return nil, ErrSize
	}
	return &Gray16{W: w, H: h, Pix: make([]uint16, w*h)}, nil
}

// FieldError 报出非法输入所在的字段名。
type FieldError struct {
	Field string
	Msg   string
}

func (e *FieldError) Error() string {
	return fmt.Sprintf("字段 %s: %s", e.Field, e.Msg)
}

// Tap 是扩散核的一拍：相对当前像素的偏移与权重。
type Tap struct {
	DX, DY int
	W      float64
}

// Kernel 是误差扩散核，可配。
type Kernel struct {
	Name string
	Taps []Tap
}

var (
	KernelNone = Kernel{Name: "none"}
	KernelFloydSteinberg = Kernel{Name: "floyd-steinberg", Taps: []Tap{
		{1, 0, 7.0 / 16},
		{-1, 1, 3.0 / 16}, {0, 1, 5.0 / 16}, {1, 1, 1.0 / 16},
	}}
	KernelAtkinson = Kernel{Name: "atkinson", Taps: []Tap{
		{1, 0, 1.0 / 8}, {2, 0, 1.0 / 8},
		{-1, 1, 1.0 / 8}, {0, 1, 1.0 / 8}, {1, 1, 1.0 / 8},
		{0, 2, 1.0 / 8},
	}}
)

// CompressOptions 配置位深压缩。扩散强度不由用户直给，
// 而是由 Levels 与 [MinSpacing, MaxSpacing] 反解得到。
type CompressOptions struct {
	SrcDepth   int     // 源位深，9..16
	DstDepth   int     // 目标位深，1..8 且小于源位深
	LoQuantile float64 // 直方图低分位裁剪点，[0,1]
	HiQuantile float64 // 直方图高分位裁剪点，[0,1]，须大于 LoQuantile
	Levels     int     // 目标色阶数，1..256
	MinSpacing float64 // 相邻色阶间距下限（输出色阶单位）
	MaxSpacing float64 // 相邻色阶间距上限
	Kernel     Kernel  // 扩散核
	MaxRefine  float64 // 步长最大细化倍数，0 表示默认 4
}

// CompressReport 是一次压缩的回执。
type CompressReport struct {
	Width, Height int
	ClipLo, ClipHi uint16  // 分位裁剪点（源位深单位）
	Strength      float64 // 反解所得扩散强度 α，[0,1]
	LevelsUsed    int     // 输出实际用到的色阶数
	Spacing       float64 // 实测相邻色阶间距（相邻已用色阶间隔的均值）
	Elapsed       time.Duration
}

// Compress16 把高位深灰度图压缩到 DstDepth 位（输出仍为 8 位容器）。
// 流程：直方图分位裁剪 → 归一化到目标位深 → 按反解强度做误差扩散。
func Compress16(g *Gray16, o CompressOptions) (*Gray, CompressReport, error) {
	var rep CompressReport
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return nil, rep, &FieldError{Field: "Image", Msg: "图为空"}
	}
	if o.SrcDepth < 9 || o.SrcDepth > 16 {
		return nil, rep, &FieldError{Field: "SrcDepth", Msg: "位深越界：须在 9 到 16 之间"}
	}
	if o.DstDepth < 1 || o.DstDepth > 8 || o.DstDepth >= o.SrcDepth {
		return nil, rep, &FieldError{Field: "DstDepth", Msg: "位深越界：须在 1 到 8 之间且小于源位深"}
	}
	if o.LoQuantile < 0 || o.LoQuantile > 1 {
		return nil, rep, &FieldError{Field: "LoQuantile", Msg: "分位越界：须在 0 到 1 之间"}
	}
	if o.HiQuantile < 0 || o.HiQuantile > 1 {
		return nil, rep, &FieldError{Field: "HiQuantile", Msg: "分位越界：须在 0 到 1 之间"}
	}
	if o.LoQuantile >= o.HiQuantile {
		return nil, rep, &FieldError{Field: "LoQuantile", Msg: "分位倒置：低分位须小于高分位"}
	}
	if o.Levels < 1 || o.Levels > 256 {
		return nil, rep, &FieldError{Field: "Levels", Msg: "色阶须在一到二百五十六之间"}
	}
	if o.MinSpacing <= 0 || o.MaxSpacing < o.MinSpacing {
		return nil, rep, &FieldError{Field: "MinSpacing", Msg: "间距区间非法"}
	}
	if o.MaxRefine == 0 {
		o.MaxRefine = 4
	}
	if o.MaxRefine < 1 {
		o.MaxRefine = 1
	}

	start := time.Now()
	rep.Width, rep.Height = g.W, g.H

	srcMax := int64(1)<<o.SrcDepth - 1
	hist := make([]int64, srcMax+1)
	for _, v := range g.Pix {
		if int64(v) <= srcMax {
			hist[v]++
		}
	}
	total := int64(len(g.Pix))
	loCut := int64(o.LoQuantile * float64(total))
	hiCut := int64(o.HiQuantile * float64(total))
	var cum int64
	loV, hiV := uint16(0), uint16(srcMax)
	setLo, setHi := false, false
	for v := int64(0); v <= srcMax; v++ {
		cum += hist[v]
		if !setLo && cum > loCut {
			loV, setLo = uint16(v), true
		}
		if !setHi && cum > hiCut {
			hiV, setHi = uint16(v), true
		}
	}
	rep.ClipLo, rep.ClipHi = loV, hiV

	outMax := float64(int64(1)<<o.DstDepth - 1)
	delta := outMax
	if o.Levels > 1 {
		delta = outMax / float64(o.Levels-1)
	}
	alpha, step := solveStrength(delta, o.MinSpacing, o.MaxSpacing, o.MaxRefine)
	rep.Strength = alpha

	span := float64(hiV) - float64(loV)
	errs := make([]float64, g.W*g.H)
	out, err := New(g.W, g.H)
	if err != nil {
		return nil, rep, err
	}
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			i := y*g.W + x
			var v float64
			if span == 0 {
				// 纯色等退化情形：裁剪区间为零，退回全量程归一化
				v = float64(g.Pix[i]) / float64(srcMax) * outMax
			} else {
				v = (float64(g.Pix[i]) - float64(loV)) / span * outMax
			}
			v += alpha * errs[i]
			if v < 0 {
				v = 0
			}
			if v > outMax {
				v = outMax
			}
			q := math.Round(v/step) * step
			if q < 0 {
				q = 0
			}
			if q > outMax {
				q = outMax
			}
			out.Pix[i] = uint8(math.Round(q))
			e := v - q
			for _, tp := range o.Kernel.Taps {
				nx, ny := x+tp.DX, y+tp.DY
				if nx >= 0 && nx < g.W && ny >= 0 && ny < g.H {
					errs[ny*g.W+nx] += tp.W * e
				}
			}
		}
	}

	rep.LevelsUsed, rep.Spacing = measureSpacing(out)
	rep.Elapsed = time.Since(start)
	return out, rep, nil
}

// solveStrength 由名义步长 delta（来自目标色阶数）反解扩散强度 α，
// 使有效步长 step = delta/(1+α(r-1)) 落入 [minS, maxS]；r 为最大细化倍数。
// delta 已落在区间内时 α 为 0；区间不可达时 α 截断到 [0,1]。
func solveStrength(delta, minS, maxS, r float64) (alpha, step float64) {
	if r < 1 {
		r = 1
	}
	target := delta
	if target > maxS {
		target = maxS
	}
	if target < minS {
		target = minS
	}
	if r == 1 || target <= 0 {
		return 0, delta
	}
	alpha = (delta/target - 1) / (r - 1)
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	step = delta / (1 + alpha*(r-1))
	return alpha, step
}

// measureSpacing 返回输出用到的色阶数与相邻已用色阶间隔的均值。
func measureSpacing(g *Gray) (levels int, spacing float64) {
	seen := map[uint8]bool{}
	for _, v := range g.Pix {
		seen[v] = true
	}
	uniq := make([]int, 0, len(seen))
	for v := range seen {
		uniq = append(uniq, int(v))
	}
	sort.Ints(uniq)
	if len(uniq) < 2 {
		return len(uniq), 0
	}
	sum := 0
	for i := 1; i < len(uniq); i++ {
		sum += uniq[i] - uniq[i-1]
	}
	return len(uniq), float64(sum) / float64(len(uniq)-1)
}
