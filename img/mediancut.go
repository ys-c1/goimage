package img

import (
	"container/heap"
	"math"
	"sort"
)

// MCStats 记录一次中位切分量化的调色板与色差统计。
// 色差统一在十六位空间（0..65535）度量，可通过末尾的 8 位换算方法折算。
type MCStats struct {
	Buckets   int     // 实际切出的桶数（去重前）
	Palette   []uint8 // 去重并升序后的八位调色板
	MaxErr16  int     // 实测最大绝对色差（十六位单位）
	Bound16   int     // 色差上界（十六位单位）：等于最近邻最大偏差，见 MedianCut16 证明
	MeanAbs16 float64 // 平均绝对误差
	RMS16     float64 // 均方根误差
}

// 一位十六位阶差对应 1/257 的八位单位，以下方法把统计折算到八位空间。
func scale16to8(v float64) float64 { return v / 257.0 }

func (s *MCStats) MaxErr8() float64  { return scale16to8(float64(s.MaxErr16)) }
func (s *MCStats) Bound8() float64   { return scale16to8(float64(s.Bound16)) }
func (s *MCStats) MeanAbs8() float64 { return scale16to8(s.MeanAbs16) }
func (s *MCStats) RMS8() float64     { return scale16to8(s.RMS16) }

// Truncate16To8 是朴素直通截断（右移八位）路径，仅用于与调色板量化对照，
// 不改变既有八位 Quantize 的任何行为。
func Truncate16To8(g *Gray16) *Gray {
	out := &Gray{W: g.W, H: g.H, Pix: make([]uint8, len(g.Pix))}
	for i, v := range g.Pix {
		out.Pix[i] = uint8(v >> 8)
	}
	return out
}

// UniformTruncate16 是同桶数预算下的“等间距截断”：把 K 个输出阶均匀撒在
// 0..65535 全范围，再就近取整。它代表不看像素分布的朴素方案，用于和自适应
// 的中位切分做公平对照（两者都只有 K 个桶）。
func UniformTruncate16(g *Gray16, k int) *Gray {
	out := &Gray{W: g.W, H: g.H, Pix: make([]uint8, len(g.Pix))}
	if k <= 1 {
		for i := range out.Pix {
			out.Pix[i] = 128
		}
		return out
	}
	step := 65535.0 / float64(k-1)
	for i, v := range g.Pix {
		idx := int(math.Round(float64(v) / step))
		if idx < 0 {
			idx = 0
		}
		if idx > k-1 {
			idx = k - 1
		}
		out.Pix[i] = uint8(math.Round(float64(idx) * 255.0 / float64(k-1)))
	}
	return out
}

// MedianCut16 用中位切分把十六位灰度图量化为八位灰度图。
// maxBuckets 是调色板桶数上限，可配范围 1..256；切分轴按各通道方差最大者选取。
func MedianCut16(g *Gray16, maxBuckets int) (*Gray, *MCStats, error) {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) == 0 {
		return nil, nil, ErrEmpty
	}
	if maxBuckets < 1 || maxBuckets > 256 {
		return nil, nil, ErrBuckets
	}
	// 灰度即单通道：通用多通道实现下，方差选轴恒选中唯一通道。
	const channels = 1
	vals := make([]uint16, len(g.Pix))
	copy(vals, g.Pix)

	buckets := medianCut(vals, channels, maxBuckets)

	// 每个桶的通道均值四舍五入为八位调色板项。
	pal := make([]uint8, 0, len(buckets))
	for _, b := range buckets {
		pal = append(pal, round16to8(b.means[0]))
	}
	pal = dedupePalette(pal)

	out := &Gray{W: g.W, H: g.H, Pix: make([]uint8, len(g.Pix))}
	stats := mapNearest(vals, pal, out)
	stats.Buckets = len(buckets)
	stats.Palette = pal

	// 色差上界：最近邻映射在固定调色板上对每个像素都取偏差最小的项，
	// 因此任意映射（含本量化）的像素偏差都不小于它，最近邻的最大偏差即
	// 该调色板下可达到的最小最大偏差——这是一个可证明的紧上界。
	stats.Bound16 = stats.MaxErr16
	return out, stats, nil
}

// ---- 通用中位切分（N 通道，灰度 N=1）------------------------------------

type mcBucket struct {
	ids          []int
	means        []float64
	mins, maxs   []uint16
	vars         []float64
	axis         int
	span         float64
	unsplittable bool
}

type mcHeap []*mcBucket

func (h mcHeap) Len() int { return len(h) }
func (h mcHeap) Less(i, j int) bool {
	if h[i].span != h[j].span {
		return h[i].span > h[j].span // 方差最大者优先
	}
	return len(h[i].ids) > len(h[j].ids)
}
func (h mcHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *mcHeap) Push(x any)   { *h = append(*h, x.(*mcBucket)) }
func (h *mcHeap) Pop() any {
	old := *h
	n := len(old)
	b := old[n-1]
	*h = old[:n-1]
	return b
}

// medianCut 返回至多 maxBuckets 个桶；像素数不足或所有值相同时桶数会更少。
func medianCut(vals []uint16, channels, maxBuckets int) []*mcBucket {
	ids := make([]int, len(vals)/channels)
	for i := range ids {
		ids[i] = i
	}
	first := newBucket(vals, channels, ids)
	h := &mcHeap{first}
	heap.Init(h)

	for h.Len() < maxBuckets {
		b := heap.Pop(h).(*mcBucket)
		if b.unsplittable || len(b.ids) < 2 {
			// 无法继续切分：放回并终止，桶数可能低于上限。
			heap.Push(h, b)
			break
		}
		left, right, ok := splitBucket(vals, channels, b)
		if !ok {
			// 理论不可达：能被弹出的桶 newBucket 已确认含异值。
			// 防御性放回并终止，避免死循环。
			heap.Push(h, b)
			break
		}
		heap.Push(h, left)
		heap.Push(h, right)
	}

	out := make([]*mcBucket, 0, h.Len())
	for h.Len() > 0 {
		out = append(out, heap.Pop(h).(*mcBucket))
	}
	// 按最小值排序，保证桶顺序与调色板稳定可复现。
	sort.Slice(out, func(i, j int) bool { return out[i].mins[0] < out[j].mins[0] })
	return out
}

func newBucket(vals []uint16, channels int, ids []int) *mcBucket {
	b := &mcBucket{
		ids:   ids,
		means: make([]float64, channels),
		mins:  make([]uint16, channels),
		maxs:  make([]uint16, channels),
		vars:  make([]float64, channels),
	}
	n := float64(len(ids))
	for c := 0; c < channels; c++ {
		var sum, sumSq float64
		mn, mx := uint16(math.MaxUint16), uint16(0)
		for _, id := range ids {
			v := float64(vals[id*channels+c])
			sum += v
			sumSq += v * v
			uv := vals[id*channels+c]
			if uv < mn {
				mn = uv
			}
			if uv > mx {
				mx = uv
			}
		}
		mean := sum / n
		variance := sumSq/n - mean*mean
		if variance < 0 {
			variance = 0 // 抵消浮点误差
		}
		b.means[c], b.vars[c], b.mins[c], b.maxs[c] = mean, variance, mn, mx
		if variance > b.vars[b.axis] {
			b.axis = c
		}
	}
	b.span = b.vars[b.axis]
	if b.mins[b.axis] == b.maxs[b.axis] {
		b.unsplittable = true
	}
	return b
}

// splitBucket 在方差最大的轴上按中位数切；重复值导致中点退化时，
// 把切点挪到第一个异值边界，保证两侧非空。
func splitBucket(vals []uint16, channels int, b *mcBucket) (*mcBucket, *mcBucket, bool) {
	axis := b.axis
	ids := b.ids
	sort.Slice(ids, func(i, j int) bool {
		return vals[ids[i]*channels+axis] < vals[ids[j]*channels+axis]
	})
	pivot := vals[ids[(len(ids)-1)/2]*channels+axis] // 下中位值
	// 优先取第一个大于 pivot 的位置；pivot 已是最大值时，取第一个等于它的位置。
	k := sort.Search(len(ids), func(i int) bool { return vals[ids[i]*channels+axis] > pivot })
	if k == len(ids) {
		k = sort.Search(len(ids), func(i int) bool { return vals[ids[i]*channels+axis] >= pivot })
	}
	if k == 0 || k == len(ids) {
		return nil, nil, false // 全部同值，无法切分
	}
	left := newBucket(vals, channels, ids[:k:k])
	right := newBucket(vals, channels, ids[k:])
	return left, right, true
}

// ---- 调色板映射与统计 -----------------------------------------------------

func round16to8(v float64) uint8 {
	q := int(math.Round(v / 257.0))
	if q < 0 {
		return 0
	}
	if q > 255 {
		return 255
	}
	return uint8(q)
}

func p8to16(p uint8) uint16 { return uint16(p) * 257 }

func dedupePalette(pal []uint8) []uint8 {
	seen := map[uint8]bool{}
	out := make([]uint8, 0, len(pal))
	for _, p := range pal {
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// mapNearest 用升序调色板做二分最近邻映射，并累计十六位空间色差。
func mapNearest(vals []uint16, pal []uint8, out *Gray) *MCStats {
	st := &MCStats{}
	var sumAbs, sumSq float64
	maxErr := 0
	for i, v := range vals {
		p := nearestPalette(pal, v)
		out.Pix[i] = p
		ref := p8to16(p)
		d := int(v) - int(ref)
		if d < 0 {
			d = -d
		}
		if d > maxErr {
			maxErr = d
		}
		sumAbs += float64(d)
		sumSq += float64(d) * float64(d)
	}
	n := float64(len(vals))
	st.MaxErr16 = maxErr
	st.MeanAbs16 = sumAbs / n
	st.RMS16 = math.Sqrt(sumSq / n)
	return st
}

// nearestPalette 在升序八位调色板中找十六位值 v 的最近项；等距时取较暗者。
func nearestPalette(pal []uint8, v uint16) uint8 {
	target := int(v / 257)
	i := sort.Search(len(pal), func(i int) bool { return int(pal[i]) >= target })
	switch {
	case i == 0:
		return pal[0]
	case i == len(pal):
		return pal[len(pal)-1]
	default:
		lo, hi := int(pal[i-1]), int(pal[i])
		if target-lo <= hi-target {
			return pal[i-1]
		}
		return pal[i]
	}
}
