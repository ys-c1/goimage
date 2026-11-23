package img

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// 图块打包（图集）实现。
//
// 流程:
//  1. 由图块总面积与最长边给出方图集边长的下界 LowerBoundSide，
//     由此得到浪费率下界 LowerBoundWaste = 1 - 总面积/下界边长²。
//  2. 反解图集边长：从下界起步逐格上探，边长小于下界时直接判不可行（下界剪枝），
//     不做无信息的二分；每个候选边长先按面积降序在剩余空闲矩形中择优摆放，
//     失败再退化为纵向堆叠。
//  3. 结果逐块核验：尺寸为零、越出图集、两两重叠、总面积超过图集。
//  4. 浪费率与下界之差的保证上界由构造性可行边长 UpperBoundSide 给出。

// Rect 是图集内的整数矩形。
type Rect struct {
	X, Y, W, H int
}

// Placement 描述一个图块在图集中的落位，Block 为输入切片中的块号（从 0 起）。
type Placement struct {
	Block   int
	X, Y    int
	W, H    int
	Rotated bool
}

// Options 控制打包行为。
type Options struct {
	// AllowRotation 允许把非正方形图块旋转 90° 摆放。
	AllowRotation bool
}

// PackResult 是一次打包的完整结果与质量度量。
type PackResult struct {
	Side       int
	Atlas      *Gray
	Placements []Placement

	TotalArea int // 所有图块面积之和

	LowerBoundSide int // 方图集边长下界
	UpperBoundSide int // 构造性可行边长上界

	LowerBoundWaste float64 // 浪费率下界
	ActualWaste     float64 // 实测浪费率
	Gap             float64 // 实测浪费率 - 下界
	GapUpperBound   float64 // Gap 的保证上界
}

var (
	ErrNoTiles = errors.New("图块列表为空")
	ErrNoFit   = errors.New("在构造上界内找不到可行排布")
)

// ViolationKind 是打包校验发现的违规类别。
type ViolationKind int

const (
	ViolationZeroSize ViolationKind = iota + 1
	ViolationExceedsAtlas
	ViolationOverlap
	ViolationAtlasOverflow
	ViolationGeometry
)

func (k ViolationKind) String() string {
	switch k {
	case ViolationZeroSize:
		return "尺寸为零"
	case ViolationExceedsAtlas:
		return "图块超出图集"
	case ViolationOverlap:
		return "图块重叠"
	case ViolationAtlasOverflow:
		return "图块总面积超过图集"
	case ViolationGeometry:
		return "落位几何与图块不符"
	default:
		return "未知违规"
	}
}

// Violation 记录一次校验违规，Blocks 为相关块号（重叠时给出相互重叠的两块）。
type Violation struct {
	Kind   ViolationKind
	Blocks []int
}

func (v *Violation) Error() string {
	return fmt.Sprintf("%s: 块号 %v", v.Kind, v.Blocks)
}

// BlockNumbers 返回违规涉及的块号副本。
func (v *Violation) BlockNumbers() []int {
	return append([]int(nil), v.Blocks...)
}

// ceilSqrt 返回不小于 n 的最小整数平方根。
func ceilSqrt(n int) int {
	if n <= 0 {
		return 0
	}
	r := int(math.Sqrt(float64(n)))
	for r*r < n {
		r++
	}
	for r > 0 && (r-1)*(r-1) >= n {
		r--
	}
	return r
}

// LowerBoundSide 返回方图集边长的下界。
// 任何可行图集必须同时满足：面积 side² ≥ 总面积；每个图块放得下，即 side ≥ 最长边。
// 故二者取大即为下界。假定图块尺寸均为正（Pack 会先校验）。
func LowerBoundSide(tiles []*Gray) int {
	area, maxSide := 0, 0
	for _, t := range tiles {
		if t == nil || t.W <= 0 || t.H <= 0 {
			continue
		}
		area += t.W * t.H
		if t.W > maxSide {
			maxSide = t.W
		}
		if t.H > maxSide {
			maxSide = t.H
		}
	}
	s := ceilSqrt(area)
	if maxSide > s {
		s = maxSide
	}
	return s
}

// MaxSideGuaranteed 返回由构造保证可行的方图集边长上界：
// 把所有图块沿纵向堆叠，宽度取各自较长边（允许旋转时）或原宽，高度取较短边（或原高）。
func MaxSideGuaranteed(tiles []*Gray, opt Options) int {
	maxW, sumH := 0, 0
	for _, t := range tiles {
		if t == nil || t.W <= 0 || t.H <= 0 {
			continue
		}
		w, h := t.W, t.H
		if opt.AllowRotation && h > w {
			w, h = h, w
		}
		if w > maxW {
			maxW = w
		}
		sumH += h
	}
	if sumH > maxW {
		return sumH
	}
	return maxW
}

// LowerBoundWaste 返回浪费率 1 - 总面积/边长²。
// 边长越小该值越低，故把边长取为 LowerBoundSide 即得全局浪费率下界。
func LowerBoundWaste(totalArea, side int) float64 {
	if side <= 0 {
		return 1
	}
	return 1 - float64(totalArea)/float64(side*side)
}

// SideAdmissible 判断边长是否可能可行：任何可行图集的边长都不小于下界，
// 因此 side < lowerBound 可直接判不可行，无需尝试摆放。
func SideAdmissible(side, lowerBound int) bool {
	return side > 0 && side >= lowerBound
}

// CheckInput 在给定图集边长下检查非法输入并报出块号：
// 尺寸为零（宽或高 ≤ 0），以及图块超出图集（任一边大于图集边长）。
// atlasSide ≤ 0 表示只检查尺寸为零。
func CheckInput(tiles []*Gray, atlasSide int) []Violation {
	var zero, over []int
	for i, t := range tiles {
		if t == nil || t.W <= 0 || t.H <= 0 {
			zero = append(zero, i)
			continue
		}
		if atlasSide > 0 && (t.W > atlasSide || t.H > atlasSide) {
			over = append(over, i)
		}
	}
	var vs []Violation
	if len(zero) > 0 {
		vs = append(vs, Violation{Kind: ViolationZeroSize, Blocks: zero})
	}
	if len(over) > 0 {
		vs = append(vs, Violation{Kind: ViolationExceedsAtlas, Blocks: over})
	}
	return vs
}

// Pack 打包全部图块，默认允许 90° 旋转。
func Pack(tiles []*Gray) (*PackResult, error) {
	return PackWith(tiles, Options{AllowRotation: true})
}

// PackInto 在给定的固定图集边长下打包。若图块超出该图集，会报出对应块号。
// 边长小于下界时直接判不可行，不做摆放尝试。
func PackInto(tiles []*Gray, side int, opt Options) (*PackResult, error) {
	if len(tiles) == 0 {
		return nil, ErrNoTiles
	}
	lower := LowerBoundSide(tiles)
	if vs := CheckInput(tiles, side); len(vs) > 0 {
		return nil, &vs[0]
	}
	if vs := checkPixels(tiles); len(vs) > 0 {
		return nil, &vs[0]
	}
	if !SideAdmissible(side, lower) {
		return nil, fmt.Errorf("%w: 边长 %d 小于下界 %d", ErrNoFit, side, lower)
	}
	total := 0
	for _, t := range tiles {
		total += t.W * t.H
	}
	if total > side*side {
		all := make([]int, len(tiles))
		for i := range all {
			all[i] = i
		}
		return nil, &Violation{Kind: ViolationAtlasOverflow, Blocks: all}
	}
	order := areaOrder(tiles)
	placements, ok := greedyPack(tiles, order, side, opt)
	if !ok {
		placements, ok = stackPack(tiles, order, side, opt)
	}
	if !ok {
		return nil, ErrNoFit
	}
	if vs := Verify(side, side, tiles, placements); len(vs) > 0 {
		return nil, &vs[0]
	}
	atlas, err := render(tiles, side, placements)
	if err != nil {
		return nil, err
	}
	lowerWaste := LowerBoundWaste(total, lower)
	actualWaste := LowerBoundWaste(total, side)
	return &PackResult{
		Side:            side,
		Atlas:           atlas,
		Placements:      placements,
		TotalArea:       total,
		LowerBoundSide:  lower,
		UpperBoundSide:  side,
		LowerBoundWaste: lowerWaste,
		ActualWaste:     actualWaste,
		Gap:             actualWaste - lowerWaste,
		GapUpperBound:   actualWaste - lowerWaste,
	}, nil
}

// PackWith 打包全部图块并返回图集、落位与浪费率度量。
func PackWith(tiles []*Gray, opt Options) (*PackResult, error) {
	if len(tiles) == 0 {
		return nil, ErrNoTiles
	}
	if vs := CheckInput(tiles, 0); len(vs) > 0 {
		return nil, &vs[0]
	}
	if vs := checkPixels(tiles); len(vs) > 0 {
		return nil, &vs[0]
	}

	total := 0
	for _, t := range tiles {
		total += t.W * t.H
	}

	lower := LowerBoundSide(tiles)
	upper := MaxSideGuaranteed(tiles, opt)
	if upper < lower {
		upper = lower
	}

	order := areaOrder(tiles)

	var placements []Placement
	side := 0
	// 从下界起步：任何边长 < lower 都已被下界证明不可行，不再进入无信息的区间做二分。
	for L := lower; L <= upper; L++ {
		if ps, ok := greedyPack(tiles, order, L, opt); ok {
			placements, side = ps, L
			break
		}
		if ps, ok := stackPack(tiles, order, L, opt); ok {
			placements, side = ps, L
			break
		}
	}
	if placements == nil {
		return nil, ErrNoFit
	}

	if vs := Verify(side, side, tiles, placements); len(vs) > 0 {
		return nil, fmt.Errorf("打包结果未通过校验: %v", vs)
	}

	atlas, err := render(tiles, side, placements)
	if err != nil {
		return nil, err
	}

	lowerWaste := LowerBoundWaste(total, lower)
	actualWaste := LowerBoundWaste(total, side)
	return &PackResult{
		Side:            side,
		Atlas:           atlas,
		Placements:      placements,
		TotalArea:       total,
		LowerBoundSide:  lower,
		UpperBoundSide:  upper,
		LowerBoundWaste: lowerWaste,
		ActualWaste:     actualWaste,
		Gap:             actualWaste - lowerWaste,
		GapUpperBound:   LowerBoundWaste(total, upper) - lowerWaste,
	}, nil
}

// Verify 逐块核验落位：尺寸为零、越出图集、两两重叠、总面积超过图集。
// atlasW/atlasH 为图集尺寸；tiles 为原始图块，Violation.Blocks 索引到该切片。
func Verify(atlasW, atlasH int, tiles []*Gray, placements []Placement) []Violation {
	var vs []Violation
	var zero, over, geom []int
	for _, p := range placements {
		if p.Block < 0 || p.Block >= len(tiles) {
			geom = append(geom, p.Block)
			continue
		}
		t := tiles[p.Block]
		if t == nil || t.W <= 0 || t.H <= 0 || p.W <= 0 || p.H <= 0 {
			zero = append(zero, p.Block)
			continue
		}
		// 落位尺寸必须等于图块原始尺寸或旋转后的尺寸。
		okDims := (p.W == t.W && p.H == t.H && !p.Rotated) ||
			(p.Rotated && p.W == t.H && p.H == t.W)
		if !okDims {
			geom = append(geom, p.Block)
		}
		// 逐块检查是否越出图集。
		if p.X < 0 || p.Y < 0 || p.X+p.W > atlasW || p.Y+p.H > atlasH {
			over = append(over, p.Block)
		}
	}
	if len(zero) > 0 {
		vs = append(vs, Violation{Kind: ViolationZeroSize, Blocks: zero})
	}
	if len(over) > 0 {
		vs = append(vs, Violation{Kind: ViolationExceedsAtlas, Blocks: over})
	}
	if len(geom) > 0 {
		vs = append(vs, Violation{Kind: ViolationGeometry, Blocks: geom})
	}

	// 两两核验重叠。
	for i := 0; i < len(placements); i++ {
		a := placements[i]
		if a.W <= 0 || a.H <= 0 {
			continue
		}
		for j := i + 1; j < len(placements); j++ {
			b := placements[j]
			if b.W <= 0 || b.H <= 0 {
				continue
			}
			if rectsOverlap(a, b) {
				vs = append(vs, Violation{Kind: ViolationOverlap, Blocks: []int{a.Block, b.Block}})
			}
		}
	}

	// 总面积不得超过图集面积。
	sum := 0
	all := make([]int, 0, len(placements))
	for _, p := range placements {
		if p.W > 0 && p.H > 0 {
			sum += p.W * p.H
		}
		all = append(all, p.Block)
	}
	if sum > atlasW*atlasH {
		vs = append(vs, Violation{Kind: ViolationAtlasOverflow, Blocks: all})
	}
	return vs
}

// checkPixels 拒绝像素缓冲长度与 W×H 不符的图块，避免 render 越界读。
func checkPixels(tiles []*Gray) []Violation {
	var bad []int
	for i, t := range tiles {
		if t == nil || t.W <= 0 || t.H <= 0 {
			continue
		}
		if len(t.Pix) < t.W*t.H {
			bad = append(bad, i)
		}
	}
	if len(bad) > 0 {
		return []Violation{{Kind: ViolationGeometry, Blocks: bad}}
	}
	return nil
}

func rectsOverlap(a, b Placement) bool {
	return a.X < b.X+b.W && b.X < a.X+a.W &&
		a.Y < b.Y+b.H && b.Y < a.Y+a.H
}

type orient struct {
	w, h int
	rot  bool
}

func orientations(t *Gray, opt Options) []orient {
	out := []orient{{w: t.W, h: t.H}}
	if opt.AllowRotation && t.W != t.H {
		out = append(out, orient{w: t.H, h: t.W, rot: true})
	}
	return out
}

// areaOrder 返回按面积降序的块号；面积相同按最长边降序，再按块号保证确定性。
func areaOrder(tiles []*Gray) []int {
	order := make([]int, len(tiles))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, b := tiles[order[i]], tiles[order[j]]
		if aa, ab := a.W*a.H, b.W*b.H; aa != ab {
			return aa > ab
		}
		if ma, mb := longest(a), longest(b); ma != mb {
			return ma > mb
		}
		return order[i] < order[j]
	})
	return order
}

func longest(t *Gray) int {
	if t.W > t.H {
		return t.W
	}
	return t.H
}

// greedyPack 按面积降序，在剩余空闲矩形中择优（short-side fit）摆放。
// 返回 false 表示该边长下贪婪排布失败（不代表数学上不可行）。
func greedyPack(tiles []*Gray, order []int, side int, opt Options) ([]Placement, bool) {
	free := []Rect{{0, 0, side, side}}
	placements := make([]Placement, 0, len(tiles))

	for _, idx := range order {
		t := tiles[idx]
		found := false
		var best Placement
		var bestScore rectScore
		for _, f := range free {
			for _, o := range orientations(t, opt) {
				if o.w > f.W || o.h > f.H {
					continue
				}
				sc := rectScore{
					short: minInt(f.W-o.w, f.H-o.h),
					long:  maxInt(f.W-o.w, f.H-o.h),
					waste: f.W*f.H - o.w*o.h,
					y:     f.Y,
					x:     f.X,
					rot:   o.rot,
				}
				if !found || sc.better(bestScore) {
					found = true
					bestScore = sc
					best = Placement{Block: idx, X: f.X, Y: f.Y, W: o.w, H: o.h, Rotated: o.rot}
				}
			}
		}
		if !found {
			return nil, false
		}
		placements = append(placements, best)
		free = placeAndSplit(free, best)
	}
	return placements, true
}

type rectScore struct {
	short, long, waste int
	y, x               int
	rot                bool
}

func (s rectScore) better(o rectScore) bool {
	if s.short != o.short {
		return s.short < o.short
	}
	if s.long != o.long {
		return s.long < o.long
	}
	if s.waste != o.waste {
		return s.waste < o.waste
	}
	if s.y != o.y {
		return s.y < o.y
	}
	if s.x != o.x {
		return s.x < o.x
	}
	if s.rot != o.rot {
		return !s.rot
	}
	return false
}

// placeAndSplit 从空闲矩形中扣除刚落位的块并做包含剪枝。
func placeAndSplit(free []Rect, p Placement) []Rect {
	out := make([]Rect, 0, len(free)+2)
	for _, f := range free {
		if !fitsInside(p, f) {
			out = append(out, f)
			continue
		}
		lw, lh := f.W-p.W, f.H-p.H
		// 两种切法剩余面积相同，保留能留下更多“可用短边”的那种。
		scoreA := minInt(lw, f.H) + minInt(p.W, lh)
		scoreB := minInt(f.W, lh) + minInt(lw, p.H)
		if scoreA >= scoreB {
			if lw > 0 {
				out = append(out, Rect{f.X + p.W, f.Y, lw, f.H})
			}
			if lh > 0 {
				out = append(out, Rect{f.X, f.Y + p.H, p.W, lh})
			}
		} else {
			if lh > 0 {
				out = append(out, Rect{f.X, f.Y + p.H, f.W, lh})
			}
			if lw > 0 {
				out = append(out, Rect{f.X + p.W, f.Y, lw, p.H})
			}
		}
	}
	return pruneContained(out)
}

func fitsInside(p Placement, f Rect) bool {
	return p.X >= f.X && p.Y >= f.Y && p.X+p.W <= f.X+f.W && p.Y+p.H <= f.Y+f.H
}

func pruneContained(rs []Rect) []Rect {
	out := make([]Rect, 0, len(rs))
	for i, r := range rs {
		if r.W <= 0 || r.H <= 0 {
			continue
		}
		contained := false
		for j, s := range rs {
			if i == j || s.W <= 0 || s.H <= 0 {
				continue
			}
			if contains(s, r) && (s.W > r.W || s.H > r.H || j < i) {
				contained = true
				break
			}
		}
		if !contained {
			out = append(out, r)
		}
	}
	return out
}

func contains(s, r Rect) bool {
	return r.X >= s.X && r.Y >= s.Y && r.X+r.W <= s.X+s.W && r.Y+r.H <= s.Y+s.H
}

// stackPack 把所有图块纵向堆叠；在构造上界处必然成功。
func stackPack(tiles []*Gray, order []int, side int, opt Options) ([]Placement, bool) {
	placements := make([]Placement, 0, len(tiles))
	y := 0
	for _, idx := range order {
		t := tiles[idx]
		w, h, rot := t.W, t.H, false
		if opt.AllowRotation && h > w {
			w, h, rot = h, w, true
		}
		if w > side || y+h > side {
			return nil, false
		}
		placements = append(placements, Placement{Block: idx, X: 0, Y: y, W: w, H: h, Rotated: rot})
		y += h
	}
	return placements, true
}

// render 把各图块按落位（含 90° 旋转）拷入方图集。
func render(tiles []*Gray, side int, placements []Placement) (*Gray, error) {
	atlas, err := New(side, side)
	if err != nil {
		return nil, err
	}
	for _, p := range placements {
		t := tiles[p.Block]
		for dy := 0; dy < p.H; dy++ {
			for dx := 0; dx < p.W; dx++ {
				var sx, sy int
				if p.Rotated {
					// 顺时针 90°: 目标 (dx,dy) 取源 (dy, H-1-dx)。
					sx, sy = dy, t.H-1-dx
				} else {
					sx, sy = dx, dy
				}
				atlas.Pix[(p.Y+dy)*side+p.X+dx] = t.Pix[sy*t.W+sx]
			}
		}
	}
	return atlas, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
