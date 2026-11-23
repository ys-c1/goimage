package img

func Quantize(g *Gray, levels int) (*Gray, error) {
	if levels < 1 || levels > 256 {
		return nil, ErrLevels
	}
	out, err := New(g.W, g.H)
	if err != nil {
		return nil, err
	}
	step := 255 / (levels - 1)
	if levels == 1 {
		step = 255
	}
	for i, v := range g.Pix {
		q := int(v) / step * step
		if q > 255 {
			q = 255
		}
		out.Pix[i] = uint8(q)
	}
	return out, nil
}

func LevelsUsed(g *Gray) int {
	seen := map[uint8]bool{}
	for _, v := range g.Pix {
		seen[v] = true
	}
	return len(seen)
}
