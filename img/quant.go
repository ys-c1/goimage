package img

func Quantize(g *Gray, levels int) (*Gray, error) {
	if levels < 1 || levels > 256 {
		return nil, ErrLevels
	}
	out, err := New(g.W, g.H)
	if err != nil {
		return nil, err
	}
	out.Pix = append(out.Pix[:0], g.Pix...)
	if levels < 256 {
		pal := paletteForLevels(levels)
		step := paletteStep(levels)
		for i, v := range g.Pix {
			out.Pix[i] = pal[nearestPaletteIndex(pal, float64(v), step)]
		}
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
