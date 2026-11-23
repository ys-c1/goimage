package img

import (
	"errors"
	"fmt"
	"math"
	"time"
)

// 拼接相关错误。所有块级错误都带块号（从 0 起）。
var (
	ErrNoTiles     = errors.New("图块列表为空")
	ErrTileBlank   = errors.New("图块为空")
	ErrTileSize    = errors.New("图块尺寸不一致")
	ErrOffsetRange = errors.New("接缝偏移越界")
	ErrGrid        = errors.New("图块数无法按列数整除")
)

// maxSeamShift 接缝偏移允许范围：正负半个像素。
const maxSeamShift = 0.5

// 一维最小二乘的平坦阈值：梯度能量低于该值视为同色/平坦，偏移取 0。
const flatEps = 1e-12

// SeamOffset 单条接缝的求解结果。
type SeamOffset struct {
	Orientation string  // "竖缝"（左右相邻）或 "横缝"（上下相邻）
	Index       int     // 同方向内的缝号，从 1 起
	LeftBlock   int     // 竖缝：左侧块号；横缝：上方块号
	RightBlock  int     // 竖缝：右侧块号；横缝：下方块号
	Delta       float64 // 接缝偏移（像素），正号表示后一块沿缝向下/向右滑动
	Flat        bool    // 边界平坦，最小二乘退化，偏移按 0 处理
}

// StitchReport 拼接诊断信息。
type StitchReport struct {
	Cols, Rows   int
	TileW, TileH int
	Seams        []SeamOffset
	// EInternal0：各原块内部梯度能量之和。
	EInternal0 float64
	// ESeam0：原块边界错配梯度能量（重采样前）。
	ESeam0 float64
	// EBefore：原块在拼接邻接图上的梯度能量总和 = EInternal0 + ESeam0。
	EBefore float64
	// EInternal1：成品图非接缝邻接对的梯度能量。
	EInternal1 float64
	// ESeam1：成品图接缝邻接对的梯度能量（重采样后）。
	ESeam1 float64
	// EAfter：成品图整体梯度能量 = EInternal1 + ESeam1。
	EAfter float64
	// Duration 本次拼接耗时。
	Duration time.Duration
}

// Stitch 把等尺寸灰度图块按 cols 列网格无缝拼起。
// 单块（len(tiles)==1 且 cols==1）走直通路径，输出像素与输入完全一致。
func Stitch(tiles []*Gray, cols int) (*Gray, *StitchReport, error) {
	start := time.Now()
	if len(tiles) == 0 {
		return nil, nil, ErrNoTiles
	}
	if cols <= 0 {
		return nil, nil, fmt.Errorf("列数必须为正: %d", cols)
	}
	if len(tiles)%cols != 0 {
		return nil, nil, fmt.Errorf("%w: 图块数 %d 列数 %d", ErrGrid, len(tiles), cols)
	}
	rows := len(tiles) / cols

	// 块级校验：空块、尺寸不一致，均报出块号。
	for i, t := range tiles {
		if t == nil || t.W <= 0 || t.H <= 0 || len(t.Pix) == 0 {
			return nil, nil, fmt.Errorf("%w: 块号 %d", ErrTileBlank, i)
		}
	}
	W, H := tiles[0].W, tiles[0].H
	if len(tiles[0].Pix) != W*H {
		return nil, nil, fmt.Errorf("%w: 块号 0 像素缓冲长度 %d 与尺寸 %dx%d 不符",
			ErrTileSize, len(tiles[0].Pix), W, H)
	}
	for i := 1; i < len(tiles); i++ {
		t := tiles[i]
		if t.W != W || t.H != H || len(t.Pix) != W*H {
			return nil, nil, fmt.Errorf("%w: 块号 %d 尺寸 %dx%d，基准块为 %dx%d",
				ErrTileSize, i, t.W, t.H, W, H)
		}
	}

	rep := &StitchReport{Cols: cols, Rows: rows, TileW: W, TileH: H}

	// 单块直通路径：不做任何重采样，输出保持不变。
	if len(tiles) == 1 {
		out, err := New(W, H)
		if err != nil {
			return nil, nil, err
		}
		copy(out.Pix, tiles[0].Pix)
		rep.EInternal0 = internalEnergy(tiles[0])
		rep.EInternal1 = rep.EInternal0
		rep.EBefore = rep.EInternal0
		rep.EAfter = rep.EInternal0
		rep.Duration = time.Since(start)
		return out, rep, nil
	}

	// ---- 第一步：逐条接缝做一维最小二乘 ----
	vDelta := make([]float64, cols) // vDelta[j]：第 j 条竖缝（j=1..cols-1）
	hDelta := make([]float64, rows) // hDelta[i]：第 i 条横缝（i=1..rows-1）

	for j := 1; j < cols; j++ {
		var A, B float64
		for r := 0; r < rows; r++ {
			left := tiles[r*cols+j-1]
			right := tiles[r*cols+j]
			a, b := seamNormalEquation(left, right, true, W, H)
			A += a
			B += b
		}
		flat := B < flatEps
		d := 0.0
		if !flat {
			d = A / B
		}
		if math.Abs(d) > maxSeamShift+1e-12 {
			return nil, nil, fmt.Errorf(
				"%w: 竖缝 %d（块号 %d|%d）偏移 %.4fpx 超出 ±%.1fpx",
				ErrOffsetRange, j, j-1, j, d, maxSeamShift)
		}
		// 用真实重采样 SSE 在最小二乘解与零偏移间择优，保证接缝能量不升。
		d = pickBestVertical(d, tiles, j, rows, cols, W, H)
		vDelta[j] = d
		rep.Seams = append(rep.Seams, SeamOffset{
			Orientation: "竖缝", Index: j,
			LeftBlock: j - 1, RightBlock: j,
			Delta: d, Flat: flat,
		})
	}
	for i := 1; i < rows; i++ {
		var A, B float64
		for c := 0; c < cols; c++ {
			top := tiles[(i-1)*cols+c]
			bot := tiles[i*cols+c]
			a, b := seamNormalEquation(top, bot, false, W, H)
			A += a
			B += b
		}
		flat := B < flatEps
		d := 0.0
		if !flat {
			d = A / B
		}
		if math.Abs(d) > maxSeamShift+1e-12 {
			return nil, nil, fmt.Errorf(
				"%w: 横缝 %d（块号 %d|%d）偏移 %.4fpx 超出 ±%.1fpx",
				ErrOffsetRange, i, (i-1)*cols, i*cols, d, maxSeamShift)
		}
		d = pickBestHorizontal(d, tiles, i, rows, cols, W, H)
		hDelta[i] = d
		rep.Seams = append(rep.Seams, SeamOffset{
			Orientation: "横缝", Index: i,
			LeftBlock: (i - 1) * cols, RightBlock: i * cols,
			Delta: d, Flat: flat,
		})
	}

	// ---- 第二步：按累计偏移对每块重采样并落位 ----
	out, err := New(cols*W, rows*H)
	if err != nil {
		return nil, nil, err
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			// 符号约定：δ 为右列块相对左列块沿 y 的滑动量（下滑为正），
			// 法方程解 m=c+δg 给出 δ；对齐采样点为 p-δ（δ<0 时即向下补位）。
			// 累计偏移：第 c 列承接其左侧全部竖缝的 y 向滑动，行同理。
			var sy, sx float64
			for j := 1; j <= c; j++ {
				sy += vDelta[j]
			}
			for i := 1; i <= r; i++ {
				sx += hDelta[i]
			}
			t := tiles[r*cols+c]
			baseO := (r*H)*out.W + c*W
			for y := 0; y < H; y++ {
				for x := 0; x < W; x++ {
					out.Pix[baseO+y*out.W+x] = uint8(math.Round(
						sampleBilinear(t, float64(x)-sx, float64(y)-sy)))
				}
			}
		}
	}

	// ---- 第三步：梯度能量与不变量 ----
	rep.EInternal0 = 0
	for _, t := range tiles {
		rep.EInternal0 += internalEnergy(t)
	}
	rep.ESeam0 = seamEnergyOriginal(tiles, cols, rows, W, H)
	rep.EBefore = rep.EInternal0 + rep.ESeam0
	rep.EInternal1, rep.ESeam1 = canvasEnergies(out, cols, rows, W, H)
	rep.EAfter = rep.EInternal1 + rep.ESeam1
	rep.Duration = time.Since(start)
	return out, rep, nil
}

// seamNormalEquation 返回中心化一维最小二乘的法方程 (A, B)，解 δ=A/B。
//
// 模型：右块剖面沿缝平移 δ 后与左块匹配，m(s)=b(s)-a(s)。
// 允许一个整体色差常数 c（吸收两侧不随 s 变化的 DC 错配，例如渐变在接缝处
// 的正常取值推进），最小化 Σ(m(s)-c-δ·g(s))²。对 c 消元后只剩与梯度相关
// 的中心化错配 m'=m-mean(m)、g'=g-mean(g)：
//
//	A = Σ g'·m'，B = Σ g'²，δ = A/B。
//
// 这同时等价于最小化两侧剖面的梯度差：对模型沿 s 求导后，DC 项消失，
// 仅剩 (g_b(s)-g_a(s))-δ·g_b'(s)，与上述中心化方程同源。
// vertical=true 比较左右两块的竖缝（沿 y 的一维剖面），否则比较上下两块的横缝（沿 x）。
func seamNormalEquation(a, b *Gray, vertical bool, W, H int) (A, B float64) {
	n := H
	if !vertical {
		n = W
	}
	g := make([]float64, n)
	m := make([]float64, n)
	var gm, mm float64
	for k := 0; k < n; k++ {
		var av, bv float64
		if vertical {
			av = float64(a.Pix[k*W+(W-1)])
			bv = float64(b.Pix[k*W])
		} else {
			av = float64(a.Pix[(H-1)*W+k])
			bv = float64(b.Pix[k])
		}
		g[k] = profileGrad(b, k, vertical, W, H)
		m[k] = bv - av
		gm += g[k]
		mm += m[k]
	}
	gm /= float64(n)
	mm /= float64(n)
	for k := 0; k < n; k++ {
		gp := g[k] - gm
		mp := m[k] - mm
		A += gp * mp
		B += gp * gp
	}
	return A, B
}

// profileGrad 取块 b 边界剖面在 k 处的一阶中心差分（端点单侧差分）。
func profileGrad(b *Gray, k int, vertical bool, W, H int) float64 {
	at := func(idx int) float64 {
		if vertical {
			return float64(b.Pix[idx*W])
		}
		return float64(b.Pix[idx])
	}
	n := H
	if !vertical {
		n = W
	}
	switch {
	case n < 2:
		return 0
	case k == 0:
		return at(1) - at(0)
	case k == n-1:
		return at(n-1) - at(n-2)
	default:
		return (at(k+1) - at(k-1)) * 0.5
	}
}

// pickBestVertical 在候选偏移间按真实重采样后的竖缝 SSE 择优。
// 候选含 0，而 δ=0 时重采样值与原像素逐点相等，故选中解的接缝能量必不高于原块之和。
func pickBestVertical(ls float64, tiles []*Gray, j, rows, cols, W, H int) float64 {
	sse := func(delta float64) float64 {
		var s float64
		for r := 0; r < rows; r++ {
			left := tiles[r*cols+j-1]
			right := tiles[r*cols+j]
			for y := 0; y < H; y++ {
				lv := float64(left.Pix[y*W+(W-1)])
				rv := sampleColumn(right, float64(y)-delta) // 对齐：在 y-δ 处取右块首列
				d := rv - lv
				s += d * d
			}
		}
		return s
	}
	best, bestE := 0.0, sse(0.0)
	if math.Abs(ls) <= maxSeamShift+1e-12 {
		if e := sse(ls); e < bestE {
			best, bestE = ls, e
		}
	}
	return best
}

// pickBestHorizontal 横缝版本，候选含义同上。
func pickBestHorizontal(ls float64, tiles []*Gray, i, rows, cols, W, H int) float64 {
	sse := func(delta float64) float64 {
		var s float64
		for c := 0; c < cols; c++ {
			top := tiles[(i-1)*cols+c]
			bot := tiles[i*cols+c]
			for x := 0; x < W; x++ {
				tv := float64(top.Pix[(H-1)*W+x])
				bv := sampleRow(bot, float64(x)-delta) // 对齐：在 x-δ 处取下块首行
				d := bv - tv
				s += d * d
			}
		}
		return s
	}
	best, bestE := 0.0, sse(0.0)
	if math.Abs(ls) <= maxSeamShift+1e-12 {
		if e := sse(ls); e < bestE {
			best, bestE = ls, e
		}
	}
	return best
}

// sampleColumn 对图块首列做一维线性采样（夹取到边界）。
func sampleColumn(g *Gray, fy float64) float64 {
	return sample1D(g.W, func(idx int) float64 { return float64(g.Pix[idx*g.W]) }, fy)
}

// sampleRow 对图块首行做一维线性采样（夹取到边界）。
func sampleRow(g *Gray, fx float64) float64 {
	return sample1D(g.W, func(idx int) float64 { return float64(g.Pix[idx]) }, fx)
}

func sample1D(n int, at func(int) float64, f float64) float64 {
	if f <= 0 {
		return at(0)
	}
	if f >= float64(n-1) {
		return at(n - 1)
	}
	i := int(math.Floor(f))
	t := f - float64(i)
	return at(i)*(1-t) + at(i+1)*t
}

// sampleBilinear 二维双线性采样，坐标夹取在 [0,W-1]×[0,H-1]。
func sampleBilinear(g *Gray, fx, fy float64) float64 {
	if fx < 0 {
		fx = 0
	}
	if fy < 0 {
		fy = 0
	}
	if fx > float64(g.W-1) {
		fx = float64(g.W - 1)
	}
	if fy > float64(g.H-1) {
		fy = float64(g.H - 1)
	}
	ix := int(math.Floor(fx))
	iy := int(math.Floor(fy))
	tx := fx - float64(ix)
	ty := fy - float64(iy)
	if ix == g.W-1 {
		ix, tx = g.W-2, 1
	}
	if iy == g.H-1 {
		iy, ty = g.H-2, 1
	}
	v00 := float64(g.Pix[iy*g.W+ix])
	v10 := float64(g.Pix[iy*g.W+ix+1])
	v01 := float64(g.Pix[(iy+1)*g.W+ix])
	v11 := float64(g.Pix[(iy+1)*g.W+ix+1])
	return (v00*(1-tx)+v10*tx)*(1-ty) + (v01*(1-tx)+v11*tx)*ty
}

// internalEnergy 单块内部邻接像素梯度平方和（横向 + 纵向）。
func internalEnergy(g *Gray) float64 {
	var e float64
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			v := float64(g.Pix[y*g.W+x])
			if x+1 < g.W {
				d := float64(g.Pix[y*g.W+x+1]) - v
				e += d * d
			}
			if y+1 < g.H {
				d := float64(g.Pix[(y+1)*g.W+x]) - v
				e += d * d
			}
		}
	}
	return e
}

// seamEnergyOriginal 原块边界错配能量：竖缝取右块首列与左块末列之差，
// 横缝取下块首行与上块末行之差，逐点平方求和。
func seamEnergyOriginal(tiles []*Gray, cols, rows, W, H int) float64 {
	var e float64
	for r := 0; r < rows; r++ {
		for j := 1; j < cols; j++ {
			left := tiles[r*cols+j-1]
			right := tiles[r*cols+j]
			for y := 0; y < H; y++ {
				d := float64(right.Pix[y*W]) - float64(left.Pix[y*W+(W-1)])
				e += d * d
			}
		}
	}
	for i := 1; i < rows; i++ {
		for c := 0; c < cols; c++ {
			top := tiles[(i-1)*cols+c]
			bot := tiles[i*cols+c]
			for x := 0; x < W; x++ {
				d := float64(bot.Pix[x]) - float64(top.Pix[(H-1)*W+x])
				e += d * d
			}
		}
	}
	return e
}

// canvasEnergies 统计成品图梯度能量，区分接缝邻接对与块内邻接对。
func canvasEnergies(g *Gray, cols, rows, W, H int) (internal, seam float64) {
	isVerticalSeam := func(x int) bool { return (x+1)%W == 0 && x+1 < g.W }
	isHorizontalSeam := func(y int) bool { return (y+1)%H == 0 && y+1 < g.H }
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			v := float64(g.Pix[y*g.W+x])
			if x+1 < g.W {
				d := float64(g.Pix[y*g.W+x+1]) - v
				e := d * d
				if isVerticalSeam(x) {
					seam += e
				} else {
					internal += e
				}
			}
			if y+1 < g.H {
				d := float64(g.Pix[(y+1)*g.W+x]) - v
				e := d * d
				if isHorizontalSeam(y) {
					seam += e
				} else {
					internal += e
				}
			}
		}
	}
	return internal, seam
}
