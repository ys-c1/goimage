package img

import (
	"fmt"
)

// 系数流格式（按位打包，最高位在前）：
//
//	每块：14 位无符号整数 DC（块内像素和 0..16320）
//	      AC 按 zigzag 顺序：
//	        6 位零游程（0..62）
//	        1 位符号 + 0 阶指数哥伦布幅度码（n≥1 用 2*bitsOf(n) 位）
//	      6 位 EOB（63）表示块结束；其前的尾部零不再显式编码
//
// 该格式由 EncodeStream / DecodeStream 实现，EncodedBits 与之逐位对应。
// 全零 AC 块除 DC 与 EOB 外不占位。

type bitWriter struct {
	buf  []byte
	cur  byte
	nbit uint
}

func (w *bitWriter) bit(b int) {
	w.cur = w.cur<<1 | byte(b&1)
	w.nbit++
	if w.nbit == 8 {
		w.buf = append(w.buf, w.cur)
		w.cur, w.nbit = 0, 0
	}
}

func (w *bitWriter) bits(v, n int) {
	for i := n - 1; i >= 0; i-- {
		w.bit((v >> i) & 1)
	}
}

func (w *bitWriter) flush() []byte {
	if w.nbit > 0 {
		w.buf = append(w.buf, w.cur<<(8-w.nbit))
		w.cur, w.nbit = 0, 0
	}
	return w.buf
}

type bitReader struct {
	data []byte
	pos  int
}

func (r *bitReader) bit() (int, error) {
	if r.pos >= len(r.data)*8 {
		return 0, fmt.Errorf("%w: 提前结束", ErrStream)
	}
	b := (r.data[r.pos/8] >> (7 - uint(r.pos%8))) & 1
	r.pos++
	return int(b), nil
}

func (r *bitReader) bits(n int) (int, error) {
	v := 0
	for i := 0; i < n; i++ {
		b, err := r.bit()
		if err != nil {
			return 0, err
		}
		v = v<<1 | b
	}
	return v, nil
}

// EncodeStream 把量化系数打包为字节流。
func (d *DCT) EncodeStream() ([]byte, error) {
	if d == nil || len(d.Blocks) == 0 {
		return nil, fmt.Errorf("%w: 没有可编码的块", ErrEmpty)
	}
	var w bitWriter
	for _, blk := range d.Blocks {
		if blk.DC < 0 || blk.DC >= 1<<14 {
			return nil, fmt.Errorf("%w: 直流超出 14 位: %d", ErrStream, blk.DC)
		}
		w.bits(blk.DC, 14)
		run := 0
		for _, q := range blk.AC {
			if q == 0 {
				run++
				continue
			}
			if run > 62 {
				return nil, fmt.Errorf("%w: 零游程超出 6 位: %d", ErrStream, run)
			}
			w.bits(run, 6)
			if q < 0 {
				w.bit(1)
			} else {
				w.bit(0)
			}
			n := abs(q)
			for i := 0; i < bitsOf(n)-1; i++ {
				w.bit(0)
			}
			w.bits(n, bitsOf(n))
			run = 0
		}
		w.bits(63, 6) // EOB；尾部零游程由该标记隐含
	}
	return w.flush(), nil
}

// DecodeStream 从字节流恢复量化系数。
func DecodeStream(data []byte, w, h, step int) (*DCT, error) {
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("%w: W=%d H=%d", ErrEmpty, w, h)
	}
	if step <= 0 {
		return nil, fmt.Errorf("%w: step=%d", ErrStep, step)
	}
	bw := (w + BlockSize - 1) / BlockSize
	bh := (h + BlockSize - 1) / BlockSize
	d := &DCT{
		W: w, H: h, Step: effectiveStep(step),
		BlocksW: bw, BlocksH: bh,
		Table:  hvsTable,
		Blocks: make([]DCTBlock, 0, bw*bh),
	}
	r := &bitReader{data: data}
	for i := 0; i < bw*bh; i++ {
		dc, err := r.bits(14)
		if err != nil {
			return nil, err
		}
		var blk DCTBlock
		blk.DC = dc
		k := 0
		for {
			run, err := r.bits(6)
			if err != nil {
				return nil, err
			}
			if run == 63 {
				break
			}
			sign, err := r.bit()
			if err != nil {
				return nil, err
			}
			zeros := 0
			for {
				b, err := r.bit()
				if err != nil {
					return nil, err
				}
				if b == 1 {
					break
				}
				zeros++
				if zeros > 30 {
					return nil, fmt.Errorf("%w: 指数哥伦布码过长", ErrStream)
				}
			}
			n, err := r.bits(zeros)
			if err != nil {
				return nil, err
			}
			n |= 1 << uint(zeros)
			if sign == 1 {
				n = -n
			}
			k += run
			if k >= BlockSize*BlockSize-1 {
				return nil, fmt.Errorf("%w: 零游程越界", ErrStream)
			}
			blk.AC[k] = n
			k++
		}
		d.Blocks = append(d.Blocks, blk)
	}
	return d, nil
}
