// Package testimg 提供曝光补偿核验用的合成图对，
// 供 img 包的测试与 cmd/exposurereport 共用，避免两处用例定义漂移。
package testimg

import (
	"math"

	"goimage/img"
)

// 图对尺寸：两幅 64×32 的图，水平重叠 16 列，拼成 112×32。
const (
	PairW     = 64
	PairH     = 32
	Overlap   = 16
	SceneW    = PairW*2 - Overlap // 112
	sceneLeft = 0
	sceneMid  = PairW - Overlap // 48，图 2 首列对应的场景列
)

// Pair 是一对准拼接图：图 1 取场景左侧 PairW 列，图 2 取右侧 PairW 列，
// 中间 Overlap 列同时出现在两图中，对应同一组场景点。
type Pair struct {
	Name    string
	A, B    *img.Gray
	Overlap int
}

// texture 是场景的归一化亮度场，取值约在 [0.06, 0.94]，
// 由低频正弦叠加构成，既有平滑渐变也有细微纹理。
func texture(x, y int) float64 {
	fx, fy := float64(x), float64(y)
	return 0.5 +
		0.22*math.Sin(0.21*fx+0.07*fy) +
		0.14*math.Cos(0.13*fx-0.17*fy) +
		0.08*math.Sin(0.9*(fx+fy))
}

func roundClamp(v float64) uint8 {
	r := math.Round(v)
	if r < 0 {
		return 0
	}
	if r > 255 {
		return 255
	}
	return uint8(r)
}

// buildPair 按场景亮度函数 fA、fB 生成一对图。fA 收到图 1 的场景列，
// fB 收到图 2 的场景列（已加上 PairW-Overlap 的偏移），因此两图的重叠区
// 由同一组 sceneX 驱动。
func buildPair(name string, fA, fB func(sceneX, y int) float64) Pair {
	a, err := img.New(PairW, PairH)
	if err != nil {
		panic(err)
	}
	b, err := img.New(PairW, PairH)
	if err != nil {
		panic(err)
	}
	for y := 0; y < PairH; y++ {
		for x := 0; x < PairW; x++ {
			a.Pix[y*PairW+x] = roundClamp(fA(sceneLeft+x, y))
			b.Pix[y*PairW+x] = roundClamp(fB(sceneMid+x, y))
		}
	}
	return Pair{Name: name, A: a, B: b, Overlap: Overlap}
}

// All 返回五组核验用例：同曝光、差一档、差三档、局部过曝、纯色。
//
// 图 1 恒为基准。差一档 / 差三档分别让图 2 的进光量为图 1 的 2 倍 / 8 倍，
// 即补偿时图 2 的增益应约为 1/2 与 1/8。局部过曝让重叠区左上 16×6 的
// 场景点亮到 230，图 2 在 2 倍进光下饱和为 255，共 96 个饱和样本。
func All() []Pair {
	base := func(x, y int) float64 { return 12 + 100*texture(x, y) }
	dark := func(x, y int) float64 { return 4 + 12*texture(x, y) }
	return []Pair{
		buildPair("同曝光",
			func(x, y int) float64 { return base(x, y) },
			func(x, y int) float64 { return base(x, y) }),
		buildPair("差一档",
			func(x, y int) float64 { return base(x, y) },
			func(x, y int) float64 { return 2 * base(x, y) }),
		buildPair("差三档",
			func(x, y int) float64 { return dark(x, y) },
			func(x, y int) float64 { return 8 * dark(x, y) }),
		buildPair("局部过曝",
			func(x, y int) float64 {
				if inHotPatch(x, y) {
					return 230
				}
				return base(x, y)
			},
			func(x, y int) float64 {
				if inHotPatch(x, y) {
					return 2 * 230 // 饱和为 255
				}
				return 2 * base(x, y)
			}),
		buildPair("纯色",
			func(x, y int) float64 { return 102 },
			func(x, y int) float64 { return 204 }),
	}
}

// inHotPatch 标注重叠区内的局部过曝块：场景列 [48,64)、前 6 行。
func inHotPatch(sceneX, y int) bool {
	return sceneX >= sceneMid && sceneX < sceneMid+Overlap && y < 6
}
