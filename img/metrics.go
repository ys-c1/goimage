package img

// OutOfGamutCount 返回源图中有多少像素的目标域线性值越界（映射前的诊断）。
// 输入按源色域编码解释，输出为计数；不修改图像。
func OutOfGamutCount(im *ColorImage, cfg GamutConfig) (int, error) {
	if err := validateColorImage(im); err != nil {
		return 0, err
	}
	m, err := newGamutMapper(cfg)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, c := range im.Pix {
		if m.outOfGamut(c) {
			n++
		}
	}
	return n, nil
}

// MapQuality 汇总一次映射的量化结果，供收尾核验使用。
type MapQuality struct {
	Total int // 像素总数
	// BeforeOutOfGamut 是映射前目标域线性值越界的像素数。
	BeforeOutOfGamut int
	// ResidualOutOfGamut 是映射输出在编码前仍越界的像素数。
	// 该计数绕开最终编码 clamp，故不会被截断掩盖。
	ResidualOutOfGamut int
	// MaxHueShiftDeg 与 MaxDeltaEOK 是“源颜色 vs 映射后目标颜色”的最大值。
	MaxHueShiftDeg float64
	MaxDeltaEOK    float64
}

// MeasureMap 对源图逐像素映射并统计质量指标，返回新图与指标，不修改输入。
func MeasureMap(im *ColorImage, cfg GamutConfig) (*ColorImage, MapQuality, error) {
	var q MapQuality
	if err := validateColorImage(im); err != nil {
		return nil, q, err
	}
	m, err := newGamutMapper(cfg)
	if err != nil {
		return nil, q, err
	}
	out, err := NewColorImage(im.W, im.H)
	if err != nil {
		return nil, q, err
	}
	q.Total = len(im.Pix)
	for i, c := range im.Pix {
		if m.outOfGamut(c) {
			q.BeforeOutOfGamut++
		}
		mapped, residual := m.mapEncodedFull(c)
		out.Pix[i] = mapped
		if residual {
			q.ResidualOutOfGamut++
		}
		srcLab := m.labOfSource(c)
		outLab := m.labOfTarget(mapped)
		if hs := abs(hueShiftDeg(srcLab, outLab)); hs > q.MaxHueShiftDeg {
			q.MaxHueShiftDeg = hs
		}
		if de := deltaEOK(srcLab, outLab); de > q.MaxDeltaEOK {
			q.MaxDeltaEOK = de
		}
	}
	return out, q, nil
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
