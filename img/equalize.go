package img

import "fmt"

// FieldError 报告非法参数所在的字段名。
type FieldError struct {
	Field string
	Msg   string
}

func (e *FieldError) Error() string {
	return fmt.Sprintf("字段 %s: %s", e.Field, e.Msg)
}

const defaultLevels = 256

// EqualizeOpts 控制直方图均衡。
type EqualizeOpts struct {
	Levels      int     // 输出色阶数，0 取默认 256，合法范围 [1,256]
	ClipRatio   float64 // 裁剪限相对平均 bin 计数的比例；0 表示按一阶条件解析求解
	Passthrough bool    // 直通：输出与输入完全一致
}

// EqualizeReport 记录一次均衡的证据与反解所需信息。
type EqualizeReport struct {
	Levels            int     // 实际使用的输出色阶数
	ClipLimit         int     // 实际使用的裁剪限（bin 计数）
	ClipRatio         float64 // 实际裁剪限相对平均 bin 计数的比例
	Gain              float64 // 占用色阶上的最坏局部对比度增益
	HistAreaBefore    int     // 均衡前直方图面积（像素总数）
	HistAreaAfter     int     // 均衡后直方图面积，守恒，应等于 Before
	PixSumBefore      uint64  // 均衡前像素灰度总和（证据）
	PixSumAfter       uint64  // 均衡后像素灰度总和（证据）
	RoundTripMaxError int     // 往返（均衡再反解）实际最大误差
	RoundTripBound    int     // 往返误差上界：同一输出桶内占用的最大灰度跨度

	lut      [256]uint8 // 前向灰度映射
	occupied [256]bool  // 裁剪重分配后非空的灰度级
}

// Equalize 对 g 做带裁剪限的直方图均衡，返回结果图与证据报告。
func Equalize(g *Gray, opts EqualizeOpts) (*Gray, *EqualizeReport, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) < g.W*g.H {
		return nil, nil, &FieldError{Field: "Pix", Msg: "图为空或像素数据不足"}
	}
	levels := opts.Levels
	if levels == 0 {
		levels = defaultLevels
	}
	if levels < 1 || levels > 256 {
		return nil, nil, &FieldError{Field: "Levels", Msg: fmt.Sprintf("色阶 %d 不在 [1,256]", levels)}
	}
	n := g.W * g.H
	var hist [256]int
	var sumBefore uint64
	for _, v := range g.Pix[:n] {
		hist[v]++
		sumBefore += uint64(v)
	}
	rep := &EqualizeReport{Levels: levels, HistAreaBefore: n, PixSumBefore: sumBefore}

	out, err := New(g.W, g.H)
	if err != nil {
		return nil, nil, err
	}

	if opts.Passthrough {
		copy(out.Pix, g.Pix[:n])
		for v := 0; v < 256; v++ {
			rep.lut[v] = uint8(v)
			rep.occupied[v] = hist[v] > 0
		}
		rep.HistAreaAfter = n
		rep.PixSumAfter = sumBefore
		rep.Gain = 1
		return out, rep, nil
	}

	clip, ratio, err := clipLimit(&hist, n, opts.ClipRatio)
	if err != nil {
		return nil, nil, err
	}
	rep.ClipLimit = clip
	rep.ClipRatio = ratio

	hc := clipHist(&hist, clip)
	for v := 0; v < 256; v++ {
		rep.occupied[v] = hist[v] > 0 // 占用级以原图为准，不含重分配撒入的空级
		rep.HistAreaAfter += hc[v]
	}

	// 累积分布，固定分母 N：T(v)=(L−1)·F(v)/N，保证增益目标的分母不随裁剪变化。
	cum := 0
	var F [256]int
	for v := 0; v < 256; v++ {
		cum += hc[v]
		F[v] = cum
	}
	for v := 0; v < 256; v++ {
		if hc[v] == 0 {
			rep.lut[v] = 0
			continue
		}
		rep.lut[v] = uint8((F[v]*(levels-1) + n/2) / n)
	}

	var sumAfter uint64
	for i, v := range g.Pix[:n] {
		y := rep.lut[v]
		out.Pix[i] = y
		sumAfter += uint64(y)
	}
	rep.PixSumAfter = sumAfter

	rep.Gain = gainOf(&hc, &hist, n, levels)

	// 往返误差：上界取同一输出桶内占用灰度的最大跨度，再核对实际最大值。
	inv := inverseLUT(&rep.lut, &rep.occupied)
	var minV, maxV [256]int
	for y := 0; y < 256; y++ {
		minV[y] = -1
	}
	for v := 0; v < 256; v++ {
		if !rep.occupied[v] {
			continue
		}
		y := rep.lut[v]
		if minV[y] < 0 {
			minV[y] = v
		}
		maxV[y] = v
		d := v - int(inv[y])
		if d < 0 {
			d = -d
		}
		if d > rep.RoundTripMaxError {
			rep.RoundTripMaxError = d
		}
	}
	for y := 0; y < 256; y++ {
		if minV[y] >= 0 && maxV[y]-minV[y] > rep.RoundTripBound {
			rep.RoundTripBound = maxV[y] - minV[y]
		}
	}
	return out, rep, nil
}

// Inverse 用报告中保存的前向映射把均衡结果反解回原灰度。
func Inverse(eq *Gray, rep *EqualizeReport) (*Gray, error) {
	if eq == nil || eq.W <= 0 || eq.H <= 0 || len(eq.Pix) < eq.W*eq.H {
		return nil, &FieldError{Field: "Pix", Msg: "图为空或像素数据不足"}
	}
	inv := inverseLUT(&rep.lut, &rep.occupied)
	out, err := New(eq.W, eq.H)
	if err != nil {
		return nil, err
	}
	n := eq.W * eq.H
	for i, y := range eq.Pix[:n] {
		out.Pix[i] = inv[y]
	}
	return out, nil
}

// clipLimit 求裁剪限。ratio 为 0 时按一阶条件解析求解：
//
// 设占用色阶集 S（|S|=K），裁剪限 c，超出量 E(c)=Σ max(h(v)−c,0) 均摊回 256 个 bin，
// 裁剪后直方图 hc(v)=min(h(v),c)+E(c)/256，变换 T(v)=(L−1)·F(v)/N 的分母恒定，
// 最坏局部增益 G(c)=(L−1)/N·min_{v∈S} hc(v)。
// c ≤ h_min 时 G(c)∝c+(N−Kc)/256，dG/dc∝1−K/256 ≥ 0（K=256 取等），单调不减；
// c ≥ h_min 时 G(c)∝h_min+E(c)/256，E 单调不增，G 单调不增。
// 一阶条件 dG/dc=0 的变号点在 c=h_min，故解析最优 c*=min_{v:h(v)>0} h(v)，无需扫描。
func clipLimit(hist *[256]int, n int, ratio float64) (int, float64, error) {
	if ratio == 0 {
		c := n
		for _, h := range hist {
			if h > 0 && h < c {
				c = h
			}
		}
		return c, float64(c) * 256 / float64(n), nil
	}
	c := int(ratio*float64(n)/256 + 0.5)
	if c < 1 || c > n {
		return 0, 0, &FieldError{Field: "ClipRatio",
			Msg: fmt.Sprintf("裁剪比例 %g 越界（裁剪限须落在 [1,%d]）", ratio, n)}
	}
	return c, ratio, nil
}

// clipHist 按 c 裁剪直方图并把超出量均摊回 256 个 bin，总面积守恒。
func clipHist(hist *[256]int, c int) [256]int {
	var hc [256]int
	excess := 0
	for v := 0; v < 256; v++ {
		if hist[v] > c {
			excess += hist[v] - c
			hc[v] = c
		} else {
			hc[v] = hist[v]
		}
	}
	base, rem := excess/256, excess%256
	for v := 0; v < 256; v++ {
		hc[v] += base
		if v < rem {
			hc[v]++
		}
	}
	return hc
}

// gainOf 计算原图占用色阶上的最坏局部增益 (levels−1)·min hc(v)/N。
// 最小 bin 只在原图非空的灰度级上取，重分配撒入空级的零头不计入。
func gainOf(hc, hist *[256]int, n, levels int) float64 {
	minBin := n
	for v := 0; v < 256; v++ {
		if hist[v] > 0 && hc[v] < minBin {
			minBin = hc[v]
		}
	}
	if n <= 0 {
		return 0
	}
	return float64(levels-1) * float64(minBin) / float64(n)
}

// inverseLUT 对每个输出灰度取映射误差最小的占用灰度作为反解。
func inverseLUT(lut *[256]uint8, occupied *[256]bool) [256]uint8 {
	var inv [256]uint8
	for y := 0; y < 256; y++ {
		best, bestD := 0, 1<<30
		for v := 0; v < 256; v++ {
			if !occupied[v] {
				continue
			}
			d := int(lut[v]) - y
			if d < 0 {
				d = -d
			}
			if d < bestD {
				bestD, best = d, v
			}
		}
		inv[y] = uint8(best)
	}
	return inv
}

// GainRow 是某一裁剪比例下的增益记录。
type GainRow struct {
	Ratio     float64 // 0 表示解析最优
	ClipLimit int
	Gain      float64
}

// GainTable 对给定裁剪比例逐一计算最坏增益，用于对照解析最优。
func GainTable(g *Gray, ratios []float64) ([]GainRow, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) < g.W*g.H {
		return nil, &FieldError{Field: "Pix", Msg: "图为空或像素数据不足"}
	}
	n := g.W * g.H
	var hist [256]int
	for _, v := range g.Pix[:n] {
		hist[v]++
	}
	rows := make([]GainRow, 0, len(ratios))
	for _, r := range ratios {
		c, _, err := clipLimit(&hist, n, r)
		if err != nil {
			return nil, err
		}
		hc := clipHist(&hist, c)
		rows = append(rows, GainRow{Ratio: r, ClipLimit: c, Gain: gainOf(&hc, &hist, n, defaultLevels)})
	}
	return rows, nil
}
