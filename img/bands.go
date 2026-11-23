package img

// 环带（banding）统计。同一幅图同时按两种口径各数一遍，两口径互不替代：
//
//	口径 A（整线平坦）：整行像素同值算一条，整列像素同值也算一条，求和。
//	    它只认贯穿整幅图的平坦扫描线，是“硬环带”的严格口径。
//	口径 B（长游程）：行向和列向上，长度达到阈值的同值游程各算一条。
//	    阈值 = max(32, 长边/8)。它把不贯穿整幅图的局部平台也算进来，
//	    是“软环带”的宽松口径，能捕捉被抖动削弱的中间态。
//
// 两种口径都只做计数，不做归一化；比较时以同一图幅、同一位深的
// 逐像素量化图为基线。

const (
	// bandRunMin 是口径 B 的游程长度下限（像素）。
	bandRunMin = 32
	// bandRunDiv 决定阈值随图幅增长：长边/8，与 bandRunMin 取大。
	// 取长边的八分之一，保证小图也有足够长的平台才算环带，
	// 大图上阈值随分辨率放大，避免把抖动噪点误计成环带。
	bandRunDiv = 8
)

// BandCounts 是同一幅图上两种口径的环带计数。
type BandCounts struct {
	// ScanLines 是口径 A：整行同值线数与整列同值线数之和。
	ScanLines int
	// LongRuns 是口径 B：行向与列向上长度达到阈值的同值游程数。
	LongRuns int
}

func CountBands(g *Gray) BandCounts {
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) != g.W*g.H {
		return BandCounts{}
	}
	// 单色图（纯色、全黑）没有环带可言，两口径都记 0。
	if LevelsUsed(g) <= 1 {
		return BandCounts{}
	}
	return BandCounts{
		ScanLines: countScanLineBands(g),
		LongRuns:  countLongRuns(g),
	}
}

func countScanLineBands(g *Gray) int {
	w, h := g.W, g.H
	count := 0
	for y := 0; y < h; y++ {
		if allEqual(g.Pix[y*w : y*w+w]) {
			count++
		}
	}
	for x := 0; x < w; x++ {
		first := g.Pix[x]
		same := true
		for y := 1; y < h; y++ {
			if g.Pix[y*w+x] != first {
				same = false
				break
			}
		}
		if same {
			count++
		}
	}
	return count
}

func countLongRuns(g *Gray) int {
	w, h := g.W, g.H
	longSide := w
	if h > longSide {
		longSide = h
	}
	minLen := longSide / bandRunDiv
	if minLen < bandRunMin {
		minLen = bandRunMin
	}
	count := 0
	for y := 0; y < h; y++ {
		run := 1
		for x := 1; x <= w; x++ {
			if x < w && g.Pix[y*w+x] == g.Pix[y*w+x-1] {
				run++
				continue
			}
			if run >= minLen {
				count++
			}
			run = 1
		}
	}
	for x := 0; x < w; x++ {
		run := 1
		for y := 1; y <= h; y++ {
			if y < h && g.Pix[y*w+x] == g.Pix[(y-1)*w+x] {
				run++
				continue
			}
			if run >= minLen {
				count++
			}
			run = 1
		}
	}
	return count
}

func allEqual(p []uint8) bool {
	for _, v := range p[1:] {
		if v != p[0] {
			return false
		}
	}
	return true
}

// BandReduction 返回 (基线-抖动后)/基线 的比例；基线为 0 时 ok=false。
func BandReduction(baseline, dithered int) (ratio float64, ok bool) {
	if baseline <= 0 {
		return 0, false
	}
	return float64(baseline-dithered) / float64(baseline), true
}
