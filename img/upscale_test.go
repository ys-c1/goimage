package img_test

import (
	"errors"
	"math"
	"strconv"
	"testing"

	"goimage/img"
	"goimage/internal/testimg"
)

// 两档放大倍率。
var scales = []float64{2, 4}

func TestUpscaleNearestStable(t *testing.T) {
	g, err := img.New(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Pix {
		g.Pix[i] = uint8(10 + i*10)
	}
	// 倍率为 1 时原样返回。
	same, err := img.UpscaleNearest(g, 1)
	if err != nil {
		t.Fatal(err)
	}
	if same.W != g.W || same.H != g.H {
		t.Fatalf("倍率 1 尺寸变化 %dx%d", same.W, same.H)
	}
	for i := range same.Pix {
		if same.Pix[i] != g.Pix[i] {
			t.Fatalf("倍率 1 像素 %d 改变", i)
		}
	}
	// 整数倍率即块复制。
	up, err := img.UpscaleNearest(g, 2)
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < up.H; y++ {
		for x := 0; x < up.W; x++ {
			want := g.Pix[(y/2)*g.W+x/2]
			if up.Pix[y*up.W+x] != want {
				t.Fatalf("(%d,%d) 未按块复制", x, y)
			}
		}
	}
}

func TestUpscaleRejectsBadScale(t *testing.T) {
	g, _ := img.New(8, 8)
	for _, s := range []float64{0.5, 0, -1, math.NaN(), math.Inf(1)} {
		_, err := img.UpscaleEdgeDirected(g, s)
		assertField(t, err, "scale")
		if _, err := img.UpscaleNearest(g, s); !errors.Is(err, img.ErrScale) {
			t.Fatalf("scale=%v 最近邻未拒绝: %v", s, err)
		}
	}
}

// TestUpscaleNearestAcceptsSolid 锁定基线接口：最近邻与双三次对纯色照常放大，
// 不受边缘导向“梯度为零即拒绝”的约束。
func TestUpscaleNearestAcceptsSolid(t *testing.T) {
	var solid *img.Gray
	for _, p := range testimg.All() {
		if p.Name == "纯色" {
			solid = p.Ref
		}
	}
	for _, fn := range []func(*img.Gray, float64) (*img.Gray, error){img.UpscaleNearest, img.UpscaleBicubic} {
		out, err := fn(solid, 2)
		if err != nil {
			t.Fatalf("基线不应拒绝纯色: %v", err)
		}
		if out.Pix[0] != solid.Pix[0] {
			t.Fatalf("纯色放大后取值变化 %d", out.Pix[0])
		}
	}
}

// TestUpscaleScaleOne 倍率为 1 时输出与输入逐像素一致。
func TestUpscaleScaleOne(t *testing.T) {
	ref := testimg.All()[3].Ref // 纹理
	for _, fn := range []func(*img.Gray, float64) (*img.Gray, error){img.UpscaleNearest, img.UpscaleBicubic, img.UpscaleEdgeDirected} {
		out, err := fn(ref, 1)
		if err != nil {
			t.Fatal(err)
		}
		if d := img.DiffStats(out, ref); d.MAE != 0 || d.Max != 0 {
			t.Fatalf("倍率 1 输出不等价输入：MAE=%.3f 最大差=%d", d.MAE, d.Max)
		}
	}
}

func TestUpscaleRejectsEmptyImage(t *testing.T) {
	cases := map[string]*img.Gray{
		"nil":    nil,
		"零宽":     {W: 0, H: 4},
		"像素不足":   {W: 4, H: 4, Pix: make([]uint8, 3)},
		"零尺寸零像素": {W: 0, H: 0},
	}
	for name, g := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := img.UpscaleEdgeDirected(g, 2)
			assertField(t, err, "image")
		})
	}
}

func TestUpscaleRejectsZeroGradient(t *testing.T) {
	g, _ := img.New(16, 16)
	for i := range g.Pix {
		g.Pix[i] = 128
	}
	_, err := img.UpscaleEdgeDirected(g, 2)
	assertField(t, err, "gradient")
	if !errors.Is(err, img.ErrGradientZero) {
		t.Fatalf("未返回梯度为零错误: %v", err)
	}
}

func assertField(t *testing.T, err error, field string) {
	t.Helper()
	if err == nil {
		t.Fatalf("应拒绝并报字段 %q，实际未报错", field)
	}
	var fe *img.FieldError
	if !errors.As(err, &fe) {
		t.Fatalf("错误类型不是 FieldError: %v", err)
	}
	if fe.FieldName() != field {
		t.Fatalf("字段名 = %q，期望 %q", fe.FieldName(), field)
	}
}

// TestUpscaleQuality 逐项核验两项硬指标：
//  1. 过冲幅度为零；
//  2. 结构相似度严格高于双三次基线。
func TestUpscaleQuality(t *testing.T) {
	for _, sc := range scales {
		lo := float64(sc)
		outW := testimg.Size
		for _, p := range testimg.All() {
			if p.Name == "纯色" {
				// 纯色是“梯度恒为零”的拒绝用例，基线照常放大。
				// 此处确认基线在纯色上也不产生过冲，与上表并列。
				low := testimg.Downsample(p.Ref, int(sc))
				for _, fn := range []func(*img.Gray, float64) (*img.Gray, error){img.UpscaleNearest, img.UpscaleBicubic} {
					out, err := fn(low, lo)
					if err != nil {
						t.Fatalf("纯色：基线被拒绝 %v", err)
					}
					if o := img.OvershootStats(out, low); o.Pixels != 0 {
						t.Fatalf("纯色：基线过冲 %d 像素", o.Pixels)
					}
				}
				continue
			}
			t.Run(p.Name+"/"+strconv.Itoa(int(sc))+"x", func(t *testing.T) {
				low := testimg.Downsample(p.Ref, int(sc))
				edge, err := img.UpscaleEdgeDirected(low, lo)
				if err != nil {
					t.Fatal(err)
				}
				bic, err := img.UpscaleBicubic(low, lo)
				if err != nil {
					t.Fatal(err)
				}
				if edge.W != outW || edge.H != outW {
					t.Fatalf("输出尺寸 %dx%d，期望 %d", edge.W, edge.H, outW)
				}

				sEdge := img.SSIM(edge, p.Ref)
				sBic := img.SSIM(bic, p.Ref)
				if !(sEdge > sBic) {
					t.Fatalf("结构相似度未超过双三次基线：边缘导向 %.6f，双三次 %.6f", sEdge, sBic)
				}

				// 两条判据都必须为零：源图局部 3×3 值域（较宽、独立），
				// 以及实现内部限幅所用的 2×2 双线性足迹（最严格）。
				if o := img.OvershootStats(edge, low); o.Pixels != 0 || o.Max != 0 {
					t.Fatalf("3×3 值域判据下过冲不为零：%d 像素，最大幅度 %d", o.Pixels, o.Max)
				}
				if o := img.OvershootStatsFootprint(edge, low); o.Pixels != 0 || o.Max != 0 {
					t.Fatalf("2×2 足迹判据下过冲不为零：%d 像素，最大幅度 %d", o.Pixels, o.Max)
				}
			})
		}
	}
}

// TestUpscaleBaselineOvershoot 记录两基线的过冲，确认过冲判据确实能检出双三次的振铃。
func TestUpscaleBaselineOvershoot(t *testing.T) {
	for _, p := range testimg.All() {
		if p.Name != "直边" && p.Name != "细线" {
			continue
		}
		low := testimg.Downsample(p.Ref, 2)
		bic, err := img.UpscaleBicubic(low, 2)
		if err != nil {
			t.Fatal(err)
		}
		if img.OvershootStats(bic, low).Pixels == 0 {
			t.Fatalf("%s：双三次基线应出现过冲，判据失效", p.Name)
		}
	}
}

func TestSSIMIdentical(t *testing.T) {
	for _, p := range testimg.All() {
		if s := img.SSIM(p.Ref, p.Ref); math.Abs(s-1) > 1e-9 {
			t.Fatalf("%s 自相似度 = %.9f", p.Name, s)
		}
	}
	small, _ := img.New(4, 4)
	if !math.IsNaN(img.SSIM(testimg.All()[0].Ref, small)) {
		t.Fatal("尺寸不一致时应返回 NaN")
	}
}
