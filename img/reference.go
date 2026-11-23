package img

// 本文件是“逐像素手算”风格的独立参考实现，刻意不调用 mediancut.go 中的
// 堆、二分等任何设施：朴素的方差计算、逐次切分、暴力最近邻。
// 测试用它与生产实现逐像素对照，两者一致才能说明量化结果可信。

// refMedianCut 以最直白的方式执行中位切分，返回原始桶数与升序八位调色板。
func refMedianCut(vals []uint16, maxBuckets int) (int, []uint8) {
	if len(vals) == 0 {
		return 0, nil
	}
	buckets := [][]uint16{{}}
	buckets[0] = append(buckets[0], vals...)

	for len(buckets) < maxBuckets {
		// 找方差最大的可切桶（单通道：方差即该桶自身的方差）。
		idx, bestVar := -1, 0.0
		for i, b := range buckets {
			v := refVariance(b)
			if v > bestVar {
				bestVar, idx = v, i
			}
		}
		if idx < 0 {
			break // 所有桶都只剩同值，切不动了
		}
		b := buckets[idx]
		refSort(b)
		// 与生产实现完全同规则：下中位 pivot；先找第一个严格大于 pivot
		// 的位置，若 pivot 已是最大值，则退化为第一个等于它的位置。
		pivot := b[(len(b)-1)/2]
		k := 0
		for k < len(b) && b[k] <= pivot {
			k++
		}
		if k == len(b) {
			k = 0
			for k < len(b) && b[k] < pivot {
				k++
			}
		}
		if k == 0 || k == len(b) {
			// 理论上不会发生（方差为正意味着至少两个异值），防御性退出。
			break
		}
		left := append([]uint16(nil), b[:k]...)
		right := append([]uint16(nil), b[k:]...)
		next := make([][]uint16, 0, len(buckets)+1)
		next = append(next, buckets[:idx]...)
		next = append(next, left, right)
		next = append(next, buckets[idx+1:]...)
		buckets = next
	}

	// 桶均值 -> 四舍五入八位 -> 去重排序。
	seen := map[uint8]bool{}
	pal := []uint8{}
	for _, b := range buckets {
		sum := 0
		for _, v := range b {
			sum += int(v)
		}
		mean := float64(sum) / float64(len(b))
		p := round16to8(mean)
		if !seen[p] {
			seen[p] = true
			pal = append(pal, p)
		}
	}
	refSort8(pal)
	return len(buckets), pal
}

// refMapBruteForce 暴力扫描调色板找最近项，等距取较暗者。
func refMapBruteForce(v uint16, pal []uint8) uint8 {
	target := int(v / 257)
	best := pal[0]
	bestD := absInt(target - int(best))
	for _, p := range pal[1:] {
		d := absInt(target - int(p))
		if d < bestD || (d == bestD && p < best) {
			best, bestD = p, d
		}
	}
	return best
}

func refVariance(b []uint16) float64 {
	if len(b) < 2 {
		return 0
	}
	sum, sumSq := 0.0, 0.0
	for _, v := range b {
		x := float64(v)
		sum += x
		sumSq += x * x
	}
	n := float64(len(b))
	mean := sum / n
	v := sumSq/n - mean*mean
	if v < 0 {
		return 0
	}
	return v
}

func refSort(b []uint16) {
	// 插入排序：参考实现优先直白，不追求速度。
	for i := 1; i < len(b); i++ {
		for j := i; j > 0 && b[j-1] > b[j]; j-- {
			b[j-1], b[j] = b[j], b[j-1]
		}
	}
}

func refSort8(b []uint8) {
	for i := 1; i < len(b); i++ {
		for j := i; j > 0 && b[j-1] > b[j]; j-- {
			b[j-1], b[j] = b[j], b[j-1]
		}
	}
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
