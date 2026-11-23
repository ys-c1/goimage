package img

import (
	"math"
	"testing"
)

func pattern(w, h int) *Gray {
	g, _ := New(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g.Pix[y*w+x] = uint8((x*37 + y*53 + 11) % 256)
		}
	}
	return g
}

var allFilters = []Filter{FilterNearest, FilterBilinear, FilterBicubic}

func TestIdentityPassthrough(t *testing.T) {
	g := pattern(7, 5)
	out, err := RotateScale(g, 0, 1, FilterBicubic)
	if err != nil {
		t.Fatal(err)
	}
	if out.W != g.W || out.H != g.H {
		t.Fatalf("直通画幅被改变: %dx%d", out.W, out.H)
	}
	for i := range g.Pix {
		if out.Pix[i] != g.Pix[i] {
			t.Fatalf("直通输出第 %d 像素被改动: %d != %d", i, out.Pix[i], g.Pix[i])
		}
	}
	// 360 度同样直通
	out2, err := RotateScale(g, 360, 1, FilterNearest)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Pix {
		if out2.Pix[i] != g.Pix[i] {
			t.Fatal("360 度直通像素被改动")
		}
	}
}

func TestInvalidParams(t *testing.T) {
	g := pattern(4, 4)
	if _, err := RotateScale(nil, 10, 1, FilterBilinear); err != ErrParamImage {
		t.Fatalf("空图应报 image 参数错误, 得到 %v", err)
	}
	empty := &Gray{}
	if _, err := RotateScale(empty, 10, 1, FilterBilinear); err != ErrParamImage {
		t.Fatalf("零尺寸图应报 image 参数错误, 得到 %v", err)
	}
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := RotateScale(g, bad, 1, FilterBilinear); err != ErrParamAngle {
			t.Fatalf("非法角度 %v 应报 angle 参数错误, 得到 %v", bad, err)
		}
	}
	for _, bad := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := RotateScale(g, 10, bad, FilterBilinear); err != ErrParamScale {
			t.Fatalf("非法缩放 %v 应报 scale 参数错误, 得到 %v", bad, err)
		}
	}
	if _, err := RotateScale(g, 10, 1, Filter(99)); err != ErrParamFilter {
		t.Fatalf("未知重采样方式应报 filter 参数错误, 得到 %v", err)
	}
}

func TestKernelWidth(t *testing.T) {
	// 放大不预过滤，宽度恒为 1
	for _, f := range allFilters {
		if w := KernelWidth(f, 2.0, DefaultAliasBound); w != 1 {
			t.Fatalf("%v 放大时核宽应为 1, 得到 %v", f, w)
		}
	}
	// 压缩时宽度 >=1，且压得越狠宽度越大；混叠上界越严宽度越大
	for _, f := range allFilters {
		w75 := KernelWidth(f, 0.75, DefaultAliasBound)
		w25 := KernelWidth(f, 0.25, DefaultAliasBound)
		if w75 < 1 || w25 <= w75 {
			t.Fatalf("%v 核宽单调性错误: %v %v", f, w75, w25)
		}
		wStrict := KernelWidth(f, 0.5, 1e-4)
		if wStrict <= KernelWidth(f, 0.5, DefaultAliasBound) {
			t.Fatalf("%v 更严混叠界应得更宽的核", f)
		}
	}
}

func TestRotateScenarios(t *testing.T) {
	cases := []struct {
		name     string
		angle, s float64
	}{
		{"小角", 7, 1},
		{"大角", 137, 1},
		{"放大", 30, 1.75},
		{"缩小", 30, 0.35},
		{"贴边", 45, 1},
	}
	g := pattern(9, 7)
	for _, c := range cases {
		for _, f := range allFilters {
			out, err := RotateScale(g, c.angle, c.s, f)
			if err != nil {
				t.Fatalf("%s/%v: %v", c.name, f, err)
			}
			if out.W <= 0 || out.H <= 0 || len(out.Pix) != out.W*out.H {
				t.Fatalf("%s/%v: 输出画幅非法", c.name, f)
			}
			for i, v := range out.Pix {
				if v > 255 {
					t.Fatalf("%s/%v: 像素 %d 越界", c.name, f, i)
				}
			}
		}
	}
}

func TestEdgeClipping(t *testing.T) {
	// 贴边小图 45 度旋转：四角应为背景零，中心区域应保留内容
	g, _ := New(3, 3)
	for i := range g.Pix {
		g.Pix[i] = 200
	}
	out, err := RotateScale(g, 45, 1, FilterBilinear)
	if err != nil {
		t.Fatal(err)
	}
	if out.Pix[0] != 0 || out.Pix[out.W*out.H-1] != 0 {
		t.Fatal("旋转后边角未做零填充")
	}
	center := out.Pix[(out.H/2)*out.W+out.W/2]
	if center == 0 {
		t.Fatal("中心内容在旋转后丢失")
	}
}

func TestQuarterTurnMapping(t *testing.T) {
	// 90 度最近邻（逆时针，数学正角约定），输出 outW=H、outH=W：
	// 输入 in[y,x] 落在输出 out[x, H-1-y]（行=x，列=H-1-y），直角严格双射。
	g := pattern(4, 6)
	out, err := RotateScale(g, 90, 1, FilterNearest)
	if err != nil {
		t.Fatal(err)
	}
	if out.W != g.H || out.H != g.W {
		t.Fatalf("90 度画幅错误: %dx%d", out.W, out.H)
	}
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			want := g.Pix[y*g.W+x]
			got := out.Pix[x*out.W+(g.H-1-y)]
			if got != want {
				t.Fatalf("90 度映射 (%d,%d)->(%d,%d) 错误: %d != %d", x, y, x, g.H-1-y, got, want)
			}
		}
	}
	// 180 度：画幅与原图相同，中心像素严格翻转。
	out180, err := RotateScale(g, 180, 1, FilterNearest)
	if err != nil {
		t.Fatal(err)
	}
	if out180.W != g.W || out180.H != g.H {
		t.Fatalf("180 度画幅错误: %dx%d", out180.W, out180.H)
	}
	if got := out180.Pix[(g.H-1-g.H/2)*g.W+(g.W-1-g.W/2)]; got != g.Pix[(g.H/2)*g.W+g.W/2] {
		t.Fatalf("180 度中心映射错误: %d", got)
	}
}

// massRelDiff 返回面积修正后的总亮度相对误差 |Σout/s² − Σin| / Σin。
// 旋转用常量图，背景像素恰为零，故全输出求和即覆盖区总亮度，无需几何裁剪。
func massRelDiff(g *Gray, angle, s float64, f Filter) (float64, float64, float64) {
	buf, _, _, err := resampleFloat(g, angle, s, f, DefaultAliasBound)
	if err != nil {
		panic(err)
	}
	sumIn, sumOut := 0.0, 0.0
	for _, v := range g.Pix {
		sumIn += float64(v)
	}
	for _, v := range buf {
		sumOut += v
	}
	corrected := sumOut / (s * s)
	return math.Abs(corrected-sumIn) / sumIn, sumIn, corrected
}

func TestBrightnessConservation(t *testing.T) {
	// 两档画幅各验证一次不变量。画幅需明显大于核支撑（最近邻在严格
	// 混叠界下盒核半径可达上百像素），边缘效应占比足够小守恒检验才有意义。
	sizes := []struct{ w, h int }{{200, 150}, {400, 300}}
	for tier, sz := range sizes {
		g := pattern(sz.w, sz.h)
		lim := 0.03
		if tier == 1 {
			lim = 0.01
		}

		// 无旋转纯缩放：覆盖区即整幅画布，全和直接可比。
		for _, s := range []float64{2.0, 0.75, 0.5, 0.35} {
			for _, f := range allFilters {
				d, sumIn, corrected := massRelDiff(g, 0, s, f)
				t.Logf("档%d 纯缩放 s=%.2f %v: Σin=%.1f Σout/s²=%.1f 守恒相对误差=%.2e",
					tier+1, s, f, sumIn, corrected, d)
				// 最近邻盒核 sinc 旁瓣衰减慢，极端缩小（s<=0.4）核宽达
				// 上百像素，边缘环带占比给独立容差；其余情形用统一紧容差。
				thisLim := lim
				if f == FilterNearest && s <= 0.4 {
					thisLim = 0.05
					if tier == 1 {
						thisLim = 0.01
					}
				}
				if d > thisLim {
					t.Fatalf("画幅 %dx%d s=%v %v 守恒误差 %v 超 %v", sz.w, sz.h, s, f, d, thisLim)
				}
			}
		}

		// 任意角度旋转：常量图背景像素恒为零，全输出求和即覆盖区总亮度，
		// 面积修正后守恒。
		const c = 137
		gc, _ := New(sz.w, sz.h)
		for i := range gc.Pix {
			gc.Pix[i] = c
		}
		for _, f := range allFilters {
			d, sumIn, corrected := massRelDiff(gc, 30, 1, f)
			t.Logf("档%d 旋转30° %v: Σin=%.1f 覆盖区Σout=%.1f 守恒相对误差=%.2e",
				tier+1, f, sumIn, corrected, d)
			if d > lim {
				t.Fatalf("画幅 %dx%d 旋转 %v 守恒误差 %v 超 %v", sz.w, sz.h, f, d, lim)
			}
		}
	}
}

func TestKernelWidthReport(t *testing.T) {
	for _, s := range []float64{1.0, 0.75, 0.5, 0.25} {
		t.Logf("s=%.2f 核拉伸 w 与支撑半径 r: 最近邻 w=%.3f r=%.3f | 双线性 w=%.3f r=%.3f | 双三次 w=%.3f r=%.3f",
			s,
			KernelWidth(FilterNearest, s, DefaultAliasBound), KernelRadius(FilterNearest, s, DefaultAliasBound),
			KernelWidth(FilterBilinear, s, DefaultAliasBound), KernelRadius(FilterBilinear, s, DefaultAliasBound),
			KernelWidth(FilterBicubic, s, DefaultAliasBound), KernelRadius(FilterBicubic, s, DefaultAliasBound))
	}
}
