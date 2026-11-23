package img

import (
	"fmt"
	"math"
)

// FieldError 指明是哪个字段越界。
type FieldError struct {
	Field  string
	Reason string
}

func (e *FieldError) Error() string {
	return fmt.Sprintf("ace: 字段 %s: %s", e.Field, e.Reason)
}

// ACE 自适应对比度增强参数。
type ACE struct {
	Radius         int     // 请求的局部窗口半径(像素, >0);超过反解上限会被压回
	TargetContrast float64 // 目标局部对比度 c = σ/μ,(0,1)
	MaxGain        float64 // 增益硬上限(>=1)
}

// ACEStats 截断统计。
type ACEStats struct {
	RadiusSolved int // 由目标局部对比度反解的窗口半径上限
	RadiusUsed   int // 实际使用的半径(<= RadiusSolved)
	Overexposed  int // 理想输出超过 255 被截断的像素数,必须为零
	ClippedLow   int // 理想输出低于 0 被截断的像素数,必须为零
	BinHigh      int // 输出直方图 255 档计数
	BinLow       int // 输出直方图 0 档计数
	SpikeHigh    bool
	SpikeLow     bool
}

// ACEOutput 增强结果:增强图 + 增益图 + 截断统计。
type ACEOutput struct {
	Img   *Gray // 增强后图像
	Gain  *Gray // 增益图:增益 [0, MaxGain] 线性映射到 [0,255]
	Stats ACEStats
}

// RadiusForContrast 由目标局部对比度 ct 反解窗口半径上限。
// 窗口边长 2r+1 会把邻域对比度稀释约 1/(2r+1),要求稀释不超过 ct:
// 2r+1 <= 1/ct  =>  r <= (1/ct - 1)/2。
func RadiusForContrast(ct float64) int {
	if ct <= 0 || ct >= 1 {
		return 0
	}
	return int(math.Floor((1/ct - 1) / 2))
}

// StretchGlobal 全局线性拉伸基线(用于对照,会过曝)。
func StretchGlobal(g *Gray) *Gray {
	lo, hi := 255, 0
	for _, v := range g.Pix {
		if int(v) < lo {
			lo = int(v)
		}
		if int(v) > hi {
			hi = int(v)
		}
	}
	out := &Gray{W: g.W, H: g.H, Pix: make([]uint8, len(g.Pix))}
	if hi <= lo {
		copy(out.Pix, g.Pix)
		return out
	}
	for i, v := range g.Pix {
		out.Pix[i] = uint8((int(v) - lo) * 255 / (hi - lo))
	}
	return out
}

// EnhanceAdaptive 自适应对比度增强。
// 逐像素取局部窗口均值 μ 与标准差 σ,增益 g = TargetContrast·μ/σ,
// 再按亮度截顶:亮侧 g <= (255-μ)/(v-μ),暗侧 g <= μ/(μ-v),
// 保证输出不越界、过曝像素数为零。
func EnhanceAdaptive(g *Gray, p ACE) (*ACEOutput, error) {
	if p.Radius <= 0 {
		return nil, &FieldError{Field: "radius", Reason: fmt.Sprintf("窗口半径必须为正,得到 %d", p.Radius)}
	}
	if g == nil || g.W <= 0 || g.H <= 0 || len(g.Pix) != g.W*g.H {
		return nil, &FieldError{Field: "image", Reason: "图像为空或像素数与画幅不符"}
	}
	if p.TargetContrast <= 0 || p.TargetContrast >= 1 {
		return nil, &FieldError{Field: "targetContrast", Reason: fmt.Sprintf("目标局部对比度须在 (0,1),得到 %v", p.TargetContrast)}
	}
	if p.MaxGain < 1 {
		return nil, &FieldError{Field: "maxGain", Reason: fmt.Sprintf("增益上限须 >=1,得到 %v", p.MaxGain)}
	}

	rSolved := RadiusForContrast(p.TargetContrast)
	if rSolved < 1 {
		return nil, &FieldError{Field: "targetContrast", Reason: fmt.Sprintf("目标局部对比度 %v 过高,反解半径为零,无可用窗口", p.TargetContrast)}
	}
	r := p.Radius
	if r > rSolved {
		r = rSolved // 实际半径不得超过反解值
	}
	if maxR := (min(g.W, g.H) - 1) / 2; r > maxR {
		r = maxR
	}
	if r < 1 {
		return nil, &FieldError{Field: "image", Reason: "画幅太小,容不下半径为 1 的窗口"}
	}

	w, h := g.W, g.H
	// 积分图(和与平方和),窗口均值/方差 O(1)。
	iw := w + 1
	sum := make([]float64, (h+1)*iw)
	sq := make([]float64, (h+1)*iw)
	for y := 0; y < h; y++ {
		var rs, rs2 float64
		for x := 0; x < w; x++ {
			v := float64(g.Pix[y*w+x])
			rs += v
			rs2 += v * v
			sum[(y+1)*iw+x+1] = sum[y*iw+x+1] + rs
			sq[(y+1)*iw+x+1] = sq[y*iw+x+1] + rs2
		}
	}
	rect := func(m []float64, x0, y0, x1, y1 int) float64 { // [x0,x1) [y0,y1)
		return m[y1*iw+x1] - m[y0*iw+x1] - m[y1*iw+x0] + m[y0*iw+x0]
	}

	out := &Gray{W: w, H: h, Pix: make([]uint8, w*h)}
	gain := &Gray{W: w, H: h, Pix: make([]uint8, w*h)}
	var st ACEStats
	st.RadiusSolved = rSolved
	st.RadiusUsed = r

	for y := 0; y < h; y++ {
		y0 := max(0, y-r)
		y1 := min(h, y+r+1)
		for x := 0; x < w; x++ {
			x0 := max(0, x-r)
			x1 := min(w, x+r+1)
			n := float64((x1 - x0) * (y1 - y0))
			mu := rect(sum, x0, y0, x1, y1) / n
			vari := rect(sq, x0, y0, x1, y1)/n - mu*mu
			if vari < 0 {
				vari = 0
			}
			sigma := math.Sqrt(vari)
			if sigma < 1e-9 {
				return nil, &FieldError{Field: "variance", Reason: fmt.Sprintf("像素 (%d,%d) 处局部方差为零,无法解增益", x, y)}
			}
			v := float64(g.Pix[y*w+x])
			muC := mu
			if muC < 1 {
				muC = 1 // 防止近黑区域对比度除零
			}
			gw := p.TargetContrast * muC / sigma // 达到目标局部对比度所需增益
			if gw > p.MaxGain {
				gw = p.MaxGain
			}
			if gw < 1 {
				gw = 1 // 只增强不压缩:局部对比度已达标处保持原样
			}
			// 按亮度限制增益上限,避免过曝/欠曝截断
			if d := v - mu; d > 0 {
				if cap := (255 - mu) / d; gw > cap {
					gw = cap
				}
			} else if d < 0 {
				if cap := mu / -d; gw > cap {
					gw = cap
				}
			}
			if gw < 0 {
				gw = 0
			}
			o := mu + gw*(v-mu)
			if o > 255 {
				st.Overexposed++
				o = 255
			}
			if o < 0 {
				st.ClippedLow++
				o = 0
			}
			out.Pix[y*w+x] = uint8(math.Round(o))
			gv := gw / p.MaxGain * 255
			if gv > 255 {
				gv = 255
			}
			gain.Pix[y*w+x] = uint8(math.Round(gv))
		}
	}

	// 直方图截断尖峰检查:端档计数不得超过相邻内档最大值的 2 倍。
	var hist [256]int
	for _, v := range out.Pix {
		hist[v]++
	}
	st.BinLow, st.BinHigh = hist[0], hist[255]
	innerLo, innerHi := 0, 0
	for i := 1; i <= 5; i++ {
		innerLo = max(innerLo, hist[i])
	}
	for i := 250; i <= 254; i++ {
		innerHi = max(innerHi, hist[i])
	}
	st.SpikeLow = hist[0] > 0 && hist[0] > 2*innerLo
	st.SpikeHigh = hist[255] > 0 && hist[255] > 2*innerHi

	return &ACEOutput{Img: out, Gain: gain, Stats: st}, nil
}
