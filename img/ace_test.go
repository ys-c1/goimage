package img

import (
	"errors"
	"testing"
)

// 确定性伪随机噪声,保证测试可复现。
type lcg struct{ s uint32 }

func (l *lcg) next() uint32 {
	l.s = l.s*1664525 + 1013904223
	return l.s >> 24
}

func noise(g *Gray, amp int, seed uint32) {
	l := lcg{s: seed}
	for i := range g.Pix {
		d := int(l.next())%(2*amp+1) - amp
		v := int(g.Pix[i]) + d
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		g.Pix[i] = uint8(v)
	}
}

func sceneBacklit() *Gray { // 逆光:亮天空 + 暗前景
	g, _ := New(64, 48)
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			if y < g.H/2 {
				g.Pix[y*g.W+x] = 235
			} else {
				g.Pix[y*g.W+x] = 25
			}
		}
	}
	noise(g, 3, 1)
	return g
}

func sceneFog() *Gray { // 雾天:低对比中灰
	g, _ := New(64, 48)
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			g.Pix[y*g.W+x] = uint8(150 + (x*20)/g.W)
		}
	}
	noise(g, 4, 2)
	return g
}

func sceneNight() *Gray { // 夜景:大面积暗部 + 点光源
	g, _ := New(64, 48)
	for i := range g.Pix {
		g.Pix[i] = 15
	}
	for _, p := range [][2]int{{10, 10}, {40, 20}, {30, 40}} {
		g.Pix[p[1]*g.W+p[0]] = 220
	}
	noise(g, 2, 3)
	return g
}

func sceneFlat() *Gray { // 平坦:微起伏
	g, _ := New(64, 48)
	for i := range g.Pix {
		g.Pix[i] = 128
	}
	noise(g, 1, 4)
	return g
}

func sceneSolid() *Gray { // 纯色:方差为零
	g, _ := New(64, 48)
	for i := range g.Pix {
		g.Pix[i] = 200
	}
	return g
}

func tiers() map[string]ACE {
	return map[string]ACE{
		"保守档": {Radius: 64, TargetContrast: 0.10, MaxGain: 2.0},
		"增强档": {Radius: 64, TargetContrast: 0.05, MaxGain: 4.0},
	}
}

func fieldErr(t *testing.T, err error) *FieldError {
	t.Helper()
	var fe *FieldError
	if !errors.As(err, &fe) {
		t.Fatalf("应返回 FieldError,得到 %v", err)
	}
	return fe
}

func TestACERejectsRadiusZero(t *testing.T) {
	_, err := EnhanceAdaptive(sceneFlat(), ACE{Radius: 0, TargetContrast: 0.1, MaxGain: 2})
	if fe := fieldErr(t, err); fe.Field != "radius" {
		t.Fatalf("应点出 radius 字段,得到 %q", fe.Field)
	}
}

func TestACERejectsEmptyImage(t *testing.T) {
	_, err := EnhanceAdaptive(&Gray{}, ACE{Radius: 3, TargetContrast: 0.1, MaxGain: 2})
	if fe := fieldErr(t, err); fe.Field != "image" {
		t.Fatalf("应点出 image 字段,得到 %q", fe.Field)
	}
}

func TestACERejectsZeroVariance(t *testing.T) {
	_, err := EnhanceAdaptive(sceneSolid(), ACE{Radius: 3, TargetContrast: 0.1, MaxGain: 2})
	if fe := fieldErr(t, err); fe.Field != "variance" {
		t.Fatalf("纯色图方差为零应点出 variance 字段,得到 %q", fe.Field)
	}
}

func TestACEScenes(t *testing.T) {
	scenes := map[string]*Gray{
		"逆光": sceneBacklit(),
		"雾天": sceneFog(),
		"夜景": sceneNight(),
		"平坦": sceneFlat(),
	}
	for name, g := range scenes {
		for tier, p := range tiers() {
			res, err := EnhanceAdaptive(g, p)
			if err != nil {
				t.Fatalf("%s/%s: %v", name, tier, err)
			}
			st := res.Stats
			if st.Overexposed != 0 {
				t.Fatalf("%s/%s: 过曝像素数 %d,必须为零", name, tier, st.Overexposed)
			}
			if st.ClippedLow != 0 {
				t.Fatalf("%s/%s: 暗部截断像素数 %d,必须为零", name, tier, st.ClippedLow)
			}
			if st.SpikeHigh || st.SpikeLow {
				t.Fatalf("%s/%s: 直方图出现截断尖峰 (高=%v 低=%v)", name, tier, st.SpikeHigh, st.SpikeLow)
			}
			if st.RadiusUsed > st.RadiusSolved {
				t.Fatalf("%s/%s: 实际半径 %d 超过反解值 %d", name, tier, st.RadiusUsed, st.RadiusSolved)
			}
			if st.RadiusUsed < 1 {
				t.Fatalf("%s/%s: 实际半径 %d 无效", name, tier, st.RadiusUsed)
			}
		}
	}
}

func TestRadiusClampedToSolved(t *testing.T) {
	p := ACE{Radius: 1000, TargetContrast: 0.10, MaxGain: 2}
	res, err := EnhanceAdaptive(sceneFog(), p)
	if err != nil {
		t.Fatal(err)
	}
	want := RadiusForContrast(p.TargetContrast)
	if res.Stats.RadiusUsed != want {
		t.Fatalf("请求半径超限应压回反解值 %d,得到 %d", want, res.Stats.RadiusUsed)
	}
}
