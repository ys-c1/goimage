package img

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Gray struct {
	W, H int
	Pix  []uint8
}

var (
	ErrSize   = errors.New("画幅必须为正")
	ErrEmpty  = errors.New("空图不能导出")
	ErrLevels = errors.New("色阶必须在一到二百五十六之间")
	ErrHeader = errors.New("PGM 头非法")
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
	pos := 0
	tokens := 0
	var vals []int
	for tokens < 4 && pos < len(data) {
		for pos < len(data) && (data[pos] == ' ' || data[pos] == '\n' || data[pos] == '\r') {
			pos++
		}
		start := pos
		for pos < len(data) && data[pos] != ' ' && data[pos] != '\n' && data[pos] != '\r' {
			pos++
		}
		tok := string(data[start:pos])
		if tokens == 0 {
			if strings.TrimSpace(tok) != "P5" {
				return nil, ErrHeader
			}
		} else {
			n, err := strconv.Atoi(tok)
			if err != nil {
				return nil, ErrHeader
			}
			vals = append(vals, n)
		}
		tokens++
	}
	if len(vals) != 3 {
		return nil, ErrHeader
	}
	for pos < len(data) && (data[pos] == ' ' || data[pos] == '\n' || data[pos] == '\r') {
		pos++
	}
	w, h := vals[0], vals[1]
	g, err := New(w, h)
	if err != nil {
		return nil, err
	}
	if len(data)-pos < w*h {
		return nil, ErrHeader
	}
	copy(g.Pix, data[pos:pos+w*h])
	return g, nil
}
