package img

import (
	"errors"
	"fmt"
	"os"
)

type Gray struct {
	W, H int
	Pix  []uint8
}

var (
	ErrSize     = errors.New("画幅必须为正")
	ErrEmpty    = errors.New("空图不能导出")
	ErrLevels   = errors.New("色阶必须在一到二百五十六之间")
)

func New(w, h int) (*Gray, error) {
	if w <= 0 || h <= 0 {
		return nil, ErrSize
	}
	return &Gray{W: w, H: h, Pix: make([]uint8, w*h)}, nil
}

func (g *Gray) SavePGM(path string) error {
	if g.W == 0 || g.H == 0 {
		return ErrEmpty
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintf(f, "P5\n%d %d\n255\n", g.W, g.H)
	_, err = f.Write(g.Pix)
	return err
}

func LoadPGM(path string) (*Gray, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var w, h, maxv int
	if _, err := fmt.Sscanf(string(data), "P5 %d %d %d", &w, &h, &maxv); err != nil {
		return nil, err
	}
	header := len(fmt.Sprintf("P5\n%d %d\n%d\n", w, h, maxv))
	g, err := New(w, h)
	if err != nil {
		return nil, err
	}
	copy(g.Pix, data[header:])
	return g, nil
}
