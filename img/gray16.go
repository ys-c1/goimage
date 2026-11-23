package img

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

// Gray16 是十六位灰度图，像素按大端语义解释，取值范围 0..65535。
type Gray16 struct {
	W, H int
	Pix  []uint16
}

// ErrBuckets 表示调色板桶数非法（必须在 1..256 之间）。
var ErrBuckets = errors.New("调色板桶数必须在一到二百五十六之间")

// NewGray16 构造空的十六位灰度图；画幅非正返回 ErrSize。
func NewGray16(w, h int) (*Gray16, error) {
	if w <= 0 || h <= 0 {
		return nil, ErrSize
	}
	return &Gray16{W: w, H: h, Pix: make([]uint16, w*h)}, nil
}

// SavePGM16 以二进制 PGM（P5，maxval=65535，双字节大端）写出。
func (g *Gray16) SavePGM16(path string) error {
	if g == nil || g.W == 0 || g.H == 0 || len(g.Pix) == 0 {
		return ErrEmpty
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	if _, err := fmt.Fprintf(w, "P5\n%d %d\n65535\n", g.W, g.H); err != nil {
		return err
	}
	var buf [2]byte
	for _, v := range g.Pix {
		binary.BigEndian.PutUint16(buf[:], v)
		if _, err := w.Write(buf[:]); err != nil {
			return err
		}
	}
	return w.Flush()
}

// LoadPGM16 读取 maxval 大于 255 的二进制 PGM，像素按大端解码。
func LoadPGM16(path string) (*Gray16, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	w, h, maxval, pos, err := scanPGMHeader(data)
	if err != nil {
		return nil, err
	}
	if maxval <= 255 {
		return nil, ErrHeader
	}
	g, err := NewGray16(w, h)
	if err != nil {
		return nil, err
	}
	need := w * h * 2
	if len(data)-pos < need {
		return nil, ErrHeader
	}
	for i := 0; i < w*h; i++ {
		g.Pix[i] = binary.BigEndian.Uint16(data[pos+2*i:])
	}
	return g, nil
}

// scanPGMHeader 解析 PGM 头，返回宽、高、maxval 与像素数据起始偏移。
// 兼容注释（# 到行尾）与任意空白分隔。
func scanPGMHeader(data []byte) (w, h, maxval, pos int, err error) {
	p := 0
	skipSpaceAndComments := func() {
		for p < len(data) {
			c := data[p]
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				p++
			} else if c == '#' {
				for p < len(data) && data[p] != '\n' {
					p++
				}
			} else {
				return
			}
		}
	}
	readToken := func() string {
		start := p
		for p < len(data) {
			c := data[p]
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '#' {
				break
			}
			p++
		}
		return string(data[start:p])
	}
	readInt := func() (int, bool) {
		skipSpaceAndComments()
		tok := readToken()
		n := 0
		if len(tok) == 0 {
			return 0, false
		}
		for _, c := range tok {
			if c < '0' || c > '9' {
				return 0, false
			}
			n = n*10 + int(c-'0')
		}
		return n, true
	}

	skipSpaceAndComments()
	if readToken() != "P5" {
		return 0, 0, 0, 0, ErrHeader
	}
	ok := false
	if w, ok = readInt(); !ok {
		return 0, 0, 0, 0, ErrHeader
	}
	if h, ok = readInt(); !ok {
		return 0, 0, 0, 0, ErrHeader
	}
	if maxval, ok = readInt(); !ok || maxval <= 0 || maxval > 65535 {
		return 0, 0, 0, 0, ErrHeader
	}
	// 单个空白符后即像素数据。
	if p >= len(data) {
		return 0, 0, 0, 0, ErrHeader
	}
	p++
	return w, h, maxval, p, nil
}
