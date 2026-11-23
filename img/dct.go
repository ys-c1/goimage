package img

import (
	"errors"
	"fmt"
	"math"
)

// BlockSize 是变换与量化固定的块边长。
const BlockSize = 8

var (
	ErrBlockSize  = errors.New("块尺寸必须为八")
	ErrStep       = errors.New("量化步长必须为正")
	ErrBlockCount = errors.New("块结构与画幅不符")
	ErrStream     = errors.New("系数流非法")
)

// QuantTable 是逐系数的量化权重。直流项恒为 1，且不参与步长缩放。
type QuantTable [BlockSize][BlockSize]int

// DCTBlock 保存一个 8x8 块的量化结果。
//
// DC 是块内像素之和（整数，不是浮点系数），因此编码与反变换都不经过
// 取整；AC 是已除以有效步长的系数，按 zigzag 顺序存放。
type DCTBlock struct {
	DC int
	AC [BlockSize*BlockSize - 1]int
}

// DCT 是一幅图像的块变换表示。
//
// Step 是与调用方约定一致的量：EncodeDCT 与 DecodeStream 都按同一个
// 请求步长做 effectiveStep 归一，因此两端的重建网格相同。
type DCT struct {
	W, H    int
	Step    int
	BlocksW int
	BlocksH int
	Table   QuantTable
	Blocks  []DCTBlock
}

// effectiveStep 把请求步长向上取到 2 的幂。
//
// 量化表与步长都落在 2 的幂上，相邻步长的重建网格因此严格嵌套：
// 细网格包含粗网格，任何系数的量化误差都不会在步长变大时反而缩小。
// 这是"重建误差随步长单调不减"的结构性保证，而不是逐图碰运气。
// 步长 1 是最细网格，等价于只用量化表权重。
func effectiveStep(step int) int {
	if step <= 1 {
		return 1
	}
	e := 1
	for e < step {
		e <<= 1
	}
	return e
}

// 观察条件：屏幕上每度视角约 64 像素（约 0.5 米看 96dpi 屏幕）。
// 对比敏感度采用 Mannos–Sakrison 模型：
//
//	S(f) = 2.6 (0.0192 + 0.114 f) exp(-(0.114 f)^1.1),  f 单位：周/度。
//
// 8x8 DCT 的 (u,v) 系数对应频率 f = hypot(u,v) * ppd / (2*8) 周/度。
// 权重与敏感度成反比，并按 2 的幂取整（保证步长加倍时每个系数的
// 重建网格严格嵌套，从而误差随步长单调不减）；最低权重为 1。
const hvsPixelsPerDegree = 64.0

func csf(f float64) float64 {
	return 2.6 * (0.0192 + 0.114*f) * math.Exp(-math.Pow(0.114*f, 1.1))
}

func deriveHVSTable() QuantTable {
	peak := 0.0
	for i := 1; i <= 4096; i++ {
		if s := csf(float64(i) / 16); s > peak {
			peak = s
		}
	}
	var t QuantTable
	for row := 0; row < BlockSize; row++ {
		for col := 0; col < BlockSize; col++ {
			if row == 0 && col == 0 {
				// 直流不量化，权重固定为 1。
				t[row][col] = 1
				continue
			}
			f := math.Hypot(float64(row), float64(col)) * hvsPixelsPerDegree / (2 * BlockSize)
			w := math.Exp2(math.Round(math.Log2(peak / csf(f))))
			if w < 1 {
				w = 1
			}
			t[row][col] = int(w)
		}
	}
	return t
}

var hvsTable = deriveHVSTable()

// HVSTable 返回按人眼对比敏感度推导出的量化表（返回副本）。
func HVSTable() QuantTable { return hvsTable }

var dctBasis = func() (b [BlockSize][BlockSize]float64) {
	for u := 0; u < BlockSize; u++ {
		a := math.Sqrt(2.0 / BlockSize)
		if u == 0 {
			a = math.Sqrt(1.0 / BlockSize)
		}
		for x := 0; x < BlockSize; x++ {
			b[u][x] = a * math.Cos((2*float64(x)+1)*float64(u)*math.Pi/(2*BlockSize))
		}
	}
	return b
}()

// zigzag 以 [行, 列] 给出扫描顺序，第一项是直流。
var zigzag = [BlockSize * BlockSize][2]int{
	{0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2},
	{2, 1}, {3, 0}, {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4}, {0, 5},
	{1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {6, 0}, {5, 1}, {4, 2},
	{3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4},
	{4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4},
	{3, 5}, {2, 6}, {1, 7}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3},
	{7, 2}, {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {4, 7}, {5, 6},
	{6, 5}, {7, 4}, {7, 5}, {6, 6}, {5, 7}, {6, 7}, {7, 6}, {7, 7},
}

// EncodeDCT 把图按 8x8 块做二维余弦变换并按 HVS 表量化。
// blockSize 目前只接受 8；step 是全局量化步长，必须为正。
// 实际使用的是不小于 step 的最小 2 的幂（见 effectiveStep）。
// 尺寸不是 8 的整数倍时，右/下边缘复制补齐，解码只写回原画幅。
func EncodeDCT(g *Gray, blockSize, step int) (*DCT, error) {
	if blockSize != BlockSize {
		return nil, fmt.Errorf("%w: blockSize=%d", ErrBlockSize, blockSize)
	}
	if step <= 0 {
		return nil, fmt.Errorf("%w: step=%d", ErrStep, step)
	}
	if g == nil {
		return nil, fmt.Errorf("%w: g=nil（图为空，W/H 未定义）", ErrEmpty)
	}
	if g.W <= 0 {
		return nil, fmt.Errorf("%w: W=%d", ErrEmpty, g.W)
	}
	if g.H <= 0 {
		return nil, fmt.Errorf("%w: H=%d", ErrEmpty, g.H)
	}
	if len(g.Pix) < g.W*g.H {
		return nil, fmt.Errorf("%w: pix=%d，画幅=%dx%d", ErrEmpty, len(g.Pix), g.W, g.H)
	}

	d := &DCT{
		W:       g.W,
		H:       g.H,
		Step:    effectiveStep(step),
		BlocksW: (g.W + BlockSize - 1) / BlockSize,
		BlocksH: (g.H + BlockSize - 1) / BlockSize,
		Table:   hvsTable,
		Blocks:  make([]DCTBlock, 0, ((g.W+BlockSize-1)/BlockSize)*((g.H+BlockSize-1)/BlockSize)),
	}

	var block [BlockSize][BlockSize]float64
	for by := 0; by < d.BlocksH; by++ {
		for bx := 0; bx < d.BlocksW; bx++ {
			sum := 0
			for y := 0; y < BlockSize; y++ {
				for x := 0; x < BlockSize; x++ {
					v := int(g.at(bx*BlockSize+x, by*BlockSize+y))
					block[y][x] = float64(v)
					sum += v
				}
			}
			// 去掉块均值：AC 与直流正交，这样 DC 可以整数存储。
			mean := float64(sum) / (BlockSize * BlockSize)
			var centered [BlockSize][BlockSize]float64
			for y := 0; y < BlockSize; y++ {
				for x := 0; x < BlockSize; x++ {
					centered[y][x] = block[y][x] - mean
				}
			}
			coef := forwardDCT(centered)

			var out DCTBlock
			out.DC = sum
			for k := 1; k < BlockSize*BlockSize; k++ {
				row, col := zigzag[k][0], zigzag[k][1]
				scale := d.Step * d.Table[row][col]
				out.AC[k-1] = int(math.Round(coef[row][col] / float64(scale)))
			}
			d.Blocks = append(d.Blocks, out)
		}
	}
	return d, nil
}

// Decode 反量化并做逆余弦变换。每个块的像素和被校正回存储的 DC，
// 因此块均值与编码端逐位一致，与 step 无关。
func (d *DCT) Decode() (*Gray, error) {
	if d == nil {
		return nil, fmt.Errorf("%w: d=nil（图为空）", ErrEmpty)
	}
	if d.W <= 0 || d.H <= 0 {
		return nil, fmt.Errorf("%w: W=%d H=%d", ErrEmpty, d.W, d.H)
	}
	if d.BlocksW <= 0 || d.BlocksH <= 0 || len(d.Blocks) != d.BlocksW*d.BlocksH {
		return nil, fmt.Errorf("%w: blocks=%d，应为 %dx%d=%d",
			ErrBlockCount, len(d.Blocks), d.BlocksW, d.BlocksH, d.BlocksW*d.BlocksH)
	}
	if d.Step <= 0 {
		return nil, fmt.Errorf("%w: step=%d", ErrStep, d.Step)
	}
	out, err := New(d.W, d.H)
	if err != nil {
		return nil, err
	}
	idx := 0
	for by := 0; by < d.BlocksH; by++ {
		for bx := 0; bx < d.BlocksW; bx++ {
			blk := d.Blocks[idx]
			idx++

			var coef [BlockSize][BlockSize]float64
			coef[0][0] = float64(blk.DC)
			for k := 1; k < BlockSize*BlockSize; k++ {
				row, col := zigzag[k][0], zigzag[k][1]
				scale := d.Step * d.Table[row][col]
				coef[row][col] = float64(blk.AC[k-1]) * float64(scale)
			}
			block := inverseDCT(coef)
			pix := toPixels(block, blk.DC)
			for y := 0; y < BlockSize; y++ {
				py := by*BlockSize + y
				if py >= d.H {
					break
				}
				for x := 0; x < BlockSize; x++ {
					px := bx*BlockSize + x
					if px >= d.W {
						break
					}
					out.Pix[py*d.W+px] = pix[y][x]
				}
			}
		}
	}
	return out, nil
}

// EncodedBits 返回系数流在此包定义的编码下的位数：
// 每块 14 位整数 DC；AC 按 zigzag 扫描，6 位零游程 + 有符号指数哥伦布码，
// 块尾 6 位 EOB。该位数可实际解码，压缩比据此计算。
func (d *DCT) EncodedBits() int {
	bits := 0
	for _, b := range d.Blocks {
		bits += 14
		for _, q := range b.AC {
			if q == 0 {
				continue
			}
			bits += 6 + 2*bitsOf(abs(q))
		}
		bits += 6
	}
	return bits
}

// Ratio 返回原始 8bpp 像素位数与编码位数的比值。
func (d *DCT) Ratio() float64 {
	bits := d.EncodedBits()
	if bits == 0 {
		return 0
	}
	return float64(d.W*d.H*8) / float64(bits)
}

func (g *Gray) at(x, y int) uint8 {
	if x >= g.W {
		x = g.W - 1
	}
	if y >= g.H {
		y = g.H - 1
	}
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	return g.Pix[y*g.W+x]
}

func forwardDCT(block [BlockSize][BlockSize]float64) [BlockSize][BlockSize]float64 {
	var tmp, coef [BlockSize][BlockSize]float64
	for y := 0; y < BlockSize; y++ {
		for u := 0; u < BlockSize; u++ {
			s := 0.0
			for x := 0; x < BlockSize; x++ {
				s += block[y][x] * dctBasis[u][x]
			}
			tmp[y][u] = s
		}
	}
	for v := 0; v < BlockSize; v++ {
		for u := 0; u < BlockSize; u++ {
			s := 0.0
			for y := 0; y < BlockSize; y++ {
				s += tmp[y][u] * dctBasis[v][y]
			}
			// 存储尺度为 8 倍正交变换，使 DC 等于像素和。
			coef[v][u] = 8 * s
		}
	}
	return coef
}

func inverseDCT(coef [BlockSize][BlockSize]float64) [BlockSize][BlockSize]float64 {
	var tmp, block [BlockSize][BlockSize]float64
	for y := 0; y < BlockSize; y++ {
		for u := 0; u < BlockSize; u++ {
			s := 0.0
			for v := 0; v < BlockSize; v++ {
				s += coef[v][u] * dctBasis[v][y]
			}
			tmp[y][u] = s
		}
	}
	for y := 0; y < BlockSize; y++ {
		for x := 0; x < BlockSize; x++ {
			s := 0.0
			for u := 0; u < BlockSize; u++ {
				s += tmp[y][u] * dctBasis[u][x]
			}
			block[y][x] = s / 8
		}
	}
	return block
}

// toPixels 取整、夹到 [0,255]，再把块内像素和校正到 targetSum。
// 因为 targetSum 来自原块像素和，块直流在像素域也可逐位验证。
func toPixels(block [BlockSize][BlockSize]float64, targetSum int) [BlockSize][BlockSize]uint8 {
	var v [BlockSize][BlockSize]int
	sum := 0
	for y := 0; y < BlockSize; y++ {
		for x := 0; x < BlockSize; x++ {
			p := int(math.Round(block[y][x]))
			if p < 0 {
				p = 0
			}
			if p > 255 {
				p = 255
			}
			v[y][x] = p
			sum += p
		}
	}
	delta := targetSum - sum
	for delta != 0 {
		changed := false
		for y := 0; y < BlockSize && delta != 0; y++ {
			for x := 0; x < BlockSize && delta != 0; x++ {
				if delta > 0 && v[y][x] < 255 {
					v[y][x]++
					delta--
					changed = true
				} else if delta < 0 && v[y][x] > 0 {
					v[y][x]--
					delta++
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
	var out [BlockSize][BlockSize]uint8
	for y := 0; y < BlockSize; y++ {
		for x := 0; x < BlockSize; x++ {
			out[y][x] = uint8(v[y][x])
		}
	}
	return out
}

func bitsOf(v int) int {
	n := 0
	for v > 0 {
		n++
		v >>= 1
	}
	return n
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
