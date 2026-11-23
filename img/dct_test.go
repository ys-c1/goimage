package img

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func cases(t *testing.T) map[string]*Gray {
	t.Helper()
	m := map[string]*Gray{}
	m["纯色"], _ = New(32, 24)
	for i := range m["纯色"].Pix {
		m["纯色"].Pix[i] = 128
	}
	m["平坦"], _ = New(32, 24)
	for y := 0; y < 24; y++ {
		for x := 0; x < 32; x++ {
			m["平坦"].Pix[y*32+x] = uint8(80 + x*3 + y)
		}
	}
	m["纹理"], _ = New(32, 24)
	for y := 0; y < 24; y++ {
		for x := 0; x < 32; x++ {
			m["纹理"].Pix[y*32+x] = uint8(128 + 40*math.Sin(float64(x)/2) + 20*math.Cos(float64(y)/3))
		}
	}
	m["边缘"], _ = New(32, 24)
	for y := 0; y < 24; y++ {
		for x := 0; x < 32; x++ {
			// 斜边 + 硬阶跃，阶跃落在 x=13 与 y=7，均不对齐块边界，
			// 这样边缘块每个频率都有能量，误差才会随步长显出层次。
			var v int
			if x*2+3*y > 50 {
				v = 210
			} else {
				v = 45
			}
			m["边缘"].Pix[y*32+x] = uint8(v)
		}
	}
	m["噪点"], _ = New(32, 24)
	seed := uint32(1)
	for i := range m["噪点"].Pix {
		seed = seed*1664525 + 1013904223
		m["噪点"].Pix[i] = uint8(seed >> 24)
	}
	return m
}

func blockMSE(a, b *Gray) float64 {
	s := 0.0
	for i := range a.Pix {
		d := float64(a.Pix[i]) - float64(b.Pix[i])
		s += d * d
	}
	return s / float64(len(a.Pix))
}

func roundTrip(t *testing.T, g *Gray, step int) *Gray {
	t.Helper()
	d, err := EncodeDCT(g, 8, step)
	if err != nil {
		t.Fatal(err)
	}
	out, err := d.Decode()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestHVSTableDerived(t *testing.T) {
	tbl := HVSTable()
	if tbl[0][0] != 1 {
		t.Fatalf("直流权重应为 1，实为 %d", tbl[0][0])
	}
	min, max := 1<<30, 0
	for row := 0; row < 8; row++ {
		for col := 0; col < 8; col++ {
			w := tbl[row][col]
			if w < min {
				min = w
			}
			if w > max {
				max = w
			}
			if tbl[row][col] != tbl[col][row] {
				t.Fatalf("表应对称: (%d,%d)=%d (%d,%d)=%d", row, col, w, col, row, tbl[col][row])
			}
			if col > 0 && w < tbl[row][col-1] {
				t.Fatalf("权重应随水平频率不减: (%d,%d)", row, col)
			}
			if row > 0 && w < tbl[row-1][col] {
				t.Fatalf("权重应随垂直频率不减: (%d,%d)", row, col)
			}
			if !isPow2(w) {
				t.Fatalf("权重应为 2 的幂: (%d,%d)=%d", row, col, w)
			}
		}
	}
	if min != 1 {
		t.Fatalf("最低 AC 权重应为 1，实为 %d", min)
	}
	if max < 8 {
		t.Fatalf("高频权重应显著高于低频，最大值仅 %d", max)
	}
	// 人眼敏感度峰值在 4–8 周/度，所以 (1,0)/(0,1) 的权重应低于 (2,0)。
	if tbl[1][0] > tbl[2][0] || tbl[0][1] > tbl[0][2] {
		t.Fatal("近峰值频率的权重不应高于更低频")
	}
}

func isPow2(v int) bool { return v > 0 && v&(v-1) == 0 }

func TestRejectsBadParameters(t *testing.T) {
	g, _ := New(8, 8)
	for _, bs := range []int{0, 4, 16, -8} {
		_, err := EncodeDCT(g, bs, 8)
		if !errors.Is(err, ErrBlockSize) {
			t.Fatalf("blockSize=%d 应返回 ErrBlockSize，实为 %v", bs, err)
		}
		if !strings.Contains(err.Error(), "blockSize=") {
			t.Fatalf("错误应指出 blockSize 参数: %v", err)
		}
	}
	for _, step := range []int{0, -1, -100} {
		_, err := EncodeDCT(g, 8, step)
		if !errors.Is(err, ErrStep) {
			t.Fatalf("step=%d 应返回 ErrStep，实为 %v", step, err)
		}
		if !strings.Contains(err.Error(), "step=") {
			t.Fatalf("错误应指出 step 参数: %v", err)
		}
	}
	empty := &Gray{}
	for _, gg := range []*Gray{nil, empty, {W: 4, H: 0}, {W: 0, H: 4}, {W: 4, H: 4}} {
		_, err := EncodeDCT(gg, 8, 8)
		if !errors.Is(err, ErrEmpty) {
			t.Fatalf("空图 %+v 应返回 ErrEmpty，实为 %v", gg, err)
		}
	}
	// 元数据声称有像素但底层切片不足，也应拒绝。
	short := &Gray{W: 8, H: 8, Pix: make([]uint8, 10)}
	_, err := EncodeDCT(short, 8, 8)
	if !errors.Is(err, ErrEmpty) || !strings.Contains(err.Error(), "pix=") {
		t.Fatalf("像素不足应返回指出 pix 参数的 ErrEmpty，实为 %v", err)
	}
	// 参数检查先于内容检查：空图 + 坏块尺寸应报块尺寸。
	if _, err := EncodeDCT(nil, 4, 8); !errors.Is(err, ErrBlockSize) {
		t.Fatalf("应先报块尺寸，实为 %v", err)
	}
	if _, err := EncodeDCT(nil, 8, 0); !errors.Is(err, ErrStep) {
		t.Fatalf("应先报步长，实为 %v", err)
	}

	if _, err := (&DCT{}).Decode(); !errors.Is(err, ErrEmpty) {
		t.Fatalf("空结构 Decode 应返回 ErrEmpty，实为 %v", err)
	}
	bad := &DCT{W: 8, H: 8, Step: 1, BlocksW: 2, BlocksH: 1, Table: HVSTable()}
	if _, err := bad.Decode(); !errors.Is(err, ErrBlockCount) {
		t.Fatalf("块数不符应返回 ErrBlockCount，实为 %v", err)
	}
	if _, err := EncodeDCT(g, 8, 8); err != nil {
		t.Fatal(err)
	}
}

func TestErrorGrowsWithStep(t *testing.T) {
	steps := []int{1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48}
	for name, g := range cases(t) {
		prev := -1.0
		for _, step := range steps {
			out := roundTrip(t, g, step)
			mse := blockMSE(g, out)
			if mse+1e-12 < prev {
				t.Fatalf("%s: step=%d 误差 %.6f 低于更小步长 %.6f", name, step, mse, prev)
			}
			prev = mse
		}
	}
	// 含高频内容的用例，两档之间必须有可见的差别（否则单调性无意义）。
	// 平坦、纯色的 AC 很小，可在低步长下精确重建，不要求严格上升。
	for _, name := range []string{"纹理", "边缘", "噪点"} {
		g := cases(t)[name]
		lo := blockMSE(g, roundTrip(t, g, 8))
		hi := blockMSE(g, roundTrip(t, g, 32))
		if hi <= lo {
			t.Fatalf("%s: 大步长误差应更大: step8=%.4f step32=%.4f", name, lo, hi)
		}
	}
	// 纯色在任何步长下都应精确重建。
	g, _ := New(8, 8)
	for i := range g.Pix {
		g.Pix[i] = 200
	}
	for _, step := range []int{1, 8, 64, 255} {
		if mse := blockMSE(g, roundTrip(t, g, step)); mse != 0 {
			t.Fatalf("纯色 step=%d MSE=%.4f，应精确重建", step, mse)
		}
	}
}

func TestDCLossless(t *testing.T) {
	for name, g := range cases(t) {
		sum := 0
		for _, v := range g.Pix {
			sum += int(v)
		}
		for _, step := range []int{1, 4, 16, 64} {
			d, err := EncodeDCT(g, 8, step)
			if err != nil {
				t.Fatal(err)
			}
			out, err := d.Decode()
			if err != nil {
				t.Fatal(err)
			}
			got := 0
			for _, v := range out.Pix {
				got += int(v)
			}
			if got != sum {
				t.Fatalf("%s step=%d: 像素和 %d != %d（直流被量化）", name, step, got, sum)
			}
			// 逐块核对存储的 DC 就是原块像素和。
			for by := 0; by < d.BlocksH; by++ {
				for bx := 0; bx < d.BlocksW; bx++ {
					want := 0
					for y := 0; y < 8; y++ {
						for x := 0; x < 8; x++ {
							want += int(g.at(bx*8+x, by*8+y))
						}
					}
					if gotDC := d.Blocks[by*d.BlocksW+bx].DC; gotDC != want {
						t.Fatalf("%s step=%d 块(%d,%d): DC %d != %d", name, step, bx, by, gotDC, want)
					}
				}
			}
		}
	}
}

func TestMeanPreservedOnUniform(t *testing.T) {
	g, _ := New(16, 16)
	for i := range g.Pix {
		g.Pix[i] = 77
	}
	for _, step := range []int{1, 8, 64} {
		out := roundTrip(t, g, step)
		for i, v := range out.Pix {
			if v != 77 {
				t.Fatalf("纯色图 step=%d 第 %d 像素变成 %d", step, i, v)
			}
		}
	}
}

func TestOddDimensions(t *testing.T) {
	g, _ := New(13, 9)
	for i := range g.Pix {
		g.Pix[i] = uint8(i * 7)
	}
	d, err := EncodeDCT(g, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	if d.BlocksW != 2 || d.BlocksH != 2 {
		t.Fatalf("补齐后块数错误 %dx%d", d.BlocksW, d.BlocksH)
	}
	out, err := d.Decode()
	if err != nil {
		t.Fatal(err)
	}
	if out.W != 13 || out.H != 9 {
		t.Fatalf("画幅被改变: %dx%d", out.W, out.H)
	}
	// 边缘块用复制补齐，其 DC 应等于补齐后块的和。
	bx, by := 1, 1
	want := 0
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			want += int(g.at(bx*8+x, by*8+y))
		}
	}
	if dc := d.Blocks[by*d.BlocksW+bx].DC; dc != want {
		t.Fatalf("边缘块 DC %d != %d", dc, want)
	}
}

func TestStreamRoundTripMatchesBits(t *testing.T) {
	for name, g := range cases(t) {
		for _, step := range []int{2, 16} {
			d, err := EncodeDCT(g, 8, step)
			if err != nil {
				t.Fatal(err)
			}
			data, err := d.EncodeStream()
			if err != nil {
				t.Fatal(err)
			}
			bits := d.EncodedBits()
			if len(data) != (bits+7)/8 {
				t.Fatalf("%s step=%d: 流长 %d 字节与位数 %d 不符", name, step, len(data), bits)
			}
			back, err := DecodeStream(data, g.W, g.H, step)
			if err != nil {
				t.Fatal(err)
			}
			if back.BlocksW != d.BlocksW || back.BlocksH != d.BlocksH || len(back.Blocks) != len(d.Blocks) {
				t.Fatal("流解码块结构不一致")
			}
			for i := range d.Blocks {
				if d.Blocks[i] != back.Blocks[i] {
					t.Fatalf("%s step=%d 块 %d 系数不一致", name, step, i)
				}
			}
			out, err := back.Decode()
			if err != nil {
				t.Fatal(err)
			}
			direct, _ := d.Decode()
			if blockMSE(out, direct) != 0 {
				t.Fatalf("%s step=%d: 流解码与直接解码结果不同", name, step)
			}
			if r := d.Ratio(); r <= 0 || math.IsInf(r, 0) || math.IsNaN(r) {
				t.Fatalf("%s step=%d: 压缩比异常 %v", name, step, r)
			}
		}
	}
}

func TestStreamRejectsTruncatedOrBadStream(t *testing.T) {
	g, _ := New(8, 8)
	for i := range g.Pix {
		g.Pix[i] = uint8(i * 3)
	}
	d, err := EncodeDCT(g, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	data, err := d.EncodeStream()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeStream(data[:len(data)-2], 8, 8, 8); !errors.Is(err, ErrStream) {
		t.Fatalf("截断流应返回 ErrStream，实为 %v", err)
	}
	if _, err := DecodeStream(nil, 8, 8, 8); !errors.Is(err, ErrStream) {
		t.Fatalf("空流应返回 ErrStream，实为 %v", err)
	}
	if _, err := DecodeStream(data, 8, 8, 0); !errors.Is(err, ErrStep) {
		t.Fatalf("零步长应返回 ErrStep，实为 %v", err)
	}
	if _, err := DecodeStream(data, 0, 8, 8); !errors.Is(err, ErrEmpty) {
		t.Fatalf("零画幅应返回 ErrEmpty，实为 %v", err)
	}
}

func TestDecodeReportsBadStep(t *testing.T) {
	d := &DCT{W: 8, H: 8, Step: 0, BlocksW: 1, BlocksH: 1, Table: HVSTable(),
		Blocks: make([]DCTBlock, 1)}
	if _, err := d.Decode(); !errors.Is(err, ErrStep) {
		t.Fatalf("步长为零的 DCT 应返回 ErrStep，实为 %v", err)
	}
}

func TestQuantizeInterfaceUnchanged(t *testing.T) {
	g, _ := New(4, 1)
	g.Pix = []uint8{0, 60, 130, 250}
	out, err := Quantize(g, 4)
	if err != nil {
		t.Fatal(err)
	}
	if LevelsUsed(out) > 4 {
		t.Fatalf("色阶超限 %d", LevelsUsed(out))
	}
	if _, err := Quantize(g, 0); !errors.Is(err, ErrLevels) {
		t.Fatalf("原有直通接口行为被改变: %v", err)
	}
}
