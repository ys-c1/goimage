package img

import (
	"errors"
	"math"
	"testing"
)

func tile(w, h, seed int) *Gray {
	g, err := New(w, h)
	if err != nil {
		panic(err)
	}
	for i := range g.Pix {
		g.Pix[i] = uint8((i*7 + seed*13) % 256)
	}
	return g
}

func mustPack(t *testing.T, tiles []*Gray, opt Options) *PackResult {
	t.Helper()
	res, err := PackWith(tiles, opt)
	if err != nil {
		t.Fatalf("打包失败: %v", err)
	}
	return res
}

// checkResult 逐块核验：不越界、不重叠、总面积不超图集、度量自洽。
func checkResult(t *testing.T, res *PackResult, tiles []*Gray) {
	t.Helper()
	if vs := Verify(res.Side, res.Side, tiles, res.Placements); len(vs) > 0 {
		t.Fatalf("校验发现违规: %v", vs)
	}
	if len(res.Placements) != len(tiles) {
		t.Fatalf("落位数量 %d != 图块数量 %d", len(res.Placements), len(tiles))
	}
	sum := 0
	for _, p := range res.Placements {
		sum += p.W * p.H
		if p.X < 0 || p.Y < 0 || p.X+p.W > res.Side || p.Y+p.H > res.Side {
			t.Fatalf("块 %d 越出图集", p.Block)
		}
	}
	if sum > res.Side*res.Side {
		t.Fatalf("总面积 %d 超过图集面积 %d", sum, res.Side*res.Side)
	}
	if res.Side < res.LowerBoundSide {
		t.Fatalf("边长 %d 小于下界 %d", res.Side, res.LowerBoundSide)
	}
	wantLower := LowerBoundWaste(res.TotalArea, res.LowerBoundSide)
	if math.Abs(res.LowerBoundWaste-wantLower) > 1e-12 {
		t.Fatalf("下界浪费率不自洽")
	}
	wantActual := LowerBoundWaste(res.TotalArea, res.Side)
	if math.Abs(res.ActualWaste-wantActual) > 1e-12 {
		t.Fatalf("实测浪费率不自洽")
	}
	if math.Abs(res.Gap-(res.ActualWaste-res.LowerBoundWaste)) > 1e-12 {
		t.Fatalf("差值不自洽")
	}
	if res.Gap < -1e-12 {
		t.Fatalf("实测浪费率低于下界，下界证明有误")
	}
	if res.Gap > res.GapUpperBound+1e-12 {
		t.Fatalf("差值 %v 超过其保证上界 %v", res.Gap, res.GapUpperBound)
	}
}

func TestPackEqualTiles(t *testing.T) {
	tiles := []*Gray{tile(32, 32, 1), tile(32, 32, 2), tile(32, 32, 3), tile(32, 32, 4)}
	res := mustPack(t, tiles, Options{AllowRotation: true})
	checkResult(t, res, tiles)
	if res.LowerBoundSide != 64 || res.Side != 64 {
		t.Fatalf("等大块应恰好拼成 64x64，得到 side=%d 下界=%d", res.Side, res.LowerBoundSide)
	}
	if res.ActualWaste != 0 {
		t.Fatalf("等大块恰好铺满，浪费率应为 0，得到 %v", res.ActualWaste)
	}
}

func TestPackDisparateTiles(t *testing.T) {
	tiles := []*Gray{tile(40, 40, 1), tile(12, 7, 2), tile(25, 3, 3), tile(5, 5, 4), tile(30, 12, 5)}
	res := mustPack(t, tiles, Options{AllowRotation: true})
	checkResult(t, res, tiles)
	if res.LowerBoundSide < 40 {
		t.Fatalf("下界必须不小于最长边 40，得到 %d", res.LowerBoundSide)
	}
}

func TestPackLongStrips(t *testing.T) {
	tiles := []*Gray{tile(1, 100, 1), tile(100, 1, 2), tile(3, 80, 3)}
	res := mustPack(t, tiles, Options{AllowRotation: true})
	checkResult(t, res, tiles)
	if res.LowerBoundSide < 100 {
		t.Fatalf("长条块下界必须不小于 100，得到 %d", res.LowerBoundSide)
	}
}

func TestPackRotationHelps(t *testing.T) {
	tiles := []*Gray{tile(100, 20, 1), tile(20, 100, 2)}

	rot := mustPack(t, tiles, Options{AllowRotation: true})
	checkResult(t, rot, tiles)
	if rot.Side != 100 {
		t.Fatalf("允许旋转时应能放进 100x100，得到 %d", rot.Side)
	}
	anyRot := false
	for _, p := range rot.Placements {
		if p.Rotated {
			anyRot = true
		}
	}
	if !anyRot {
		t.Fatal("该用例应有图块被旋转")
	}

	fixed := mustPack(t, tiles, Options{AllowRotation: false})
	checkResult(t, fixed, tiles)
	if fixed.Side <= rot.Side {
		t.Fatalf("禁止旋转不应优于允许旋转: fixed=%d rot=%d", fixed.Side, rot.Side)
	}
}

func TestPackSingleTile(t *testing.T) {
	tiles := []*Gray{tile(7, 3, 9)}
	res := mustPack(t, tiles, Options{AllowRotation: true})
	checkResult(t, res, tiles)
	// 面积下界给 5，但最长边 7 才是真下界。
	if res.LowerBoundSide != 7 || res.Side != 7 {
		t.Fatalf("单块下界/边长应为 7，得到 下界=%d side=%d", res.LowerBoundSide, res.Side)
	}
	if len(res.Placements) != 1 {
		t.Fatalf("单块应只有一个落位")
	}
	p := res.Placements[0]
	if p.X != 0 || p.Y != 0 || p.W != 7 || p.H != 3 {
		t.Fatalf("单块落位错误: %+v", p)
	}
}

func TestLowerBoundSidePruning(t *testing.T) {
	// 细长块：面积下界只有 5，但最长边 7 把下界抬到 7。
	if got := LowerBoundSide([]*Gray{tile(7, 3, 0)}); got != 7 {
		t.Fatalf("下界应为 7，得到 %d", got)
	}
	if SideAdmissible(6, 7) {
		t.Fatal("边长 6 小于下界 7，应直接判不可行")
	}
	if !SideAdmissible(7, 7) {
		t.Fatal("边长等于下界不应被剪枝")
	}
	if LowerBoundWaste(21, 7) > LowerBoundWaste(21, 8) {
		t.Fatal("边长越大浪费率下界应越高")
	}
}

func TestPackRejectsZeroSizeWithBlockNumber(t *testing.T) {
	good := tile(4, 4, 1)
	zero := &Gray{W: 0, H: 5, Pix: nil}
	nilTile := (*Gray)(nil)
	tiles := []*Gray{good, zero, tile(2, 2, 2), nilTile}

	_, err := PackWith(tiles, Options{AllowRotation: true})
	if err == nil {
		t.Fatal("尺寸为零应报错")
	}
	v, ok := err.(*Violation)
	if !ok {
		t.Fatalf("错误类型应为 *Violation，得到 %T", err)
	}
	if v.Kind != ViolationZeroSize {
		t.Fatalf("违规类别应为尺寸为零，得到 %v", v.Kind)
	}
	got := v.BlockNumbers()
	if len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Fatalf("应报出块号 [1 3]，得到 %v", got)
	}
}

func TestVerifyExceedsAtlasWithBlockNumber(t *testing.T) {
	tiles := []*Gray{tile(4, 4, 1), tile(9, 2, 2)}
	placements := []Placement{
		{Block: 0, X: 0, Y: 0, W: 4, H: 4},
		{Block: 1, X: 5, Y: 0, W: 9, H: 2}, // 5+9=14 > 8
	}
	vs := Verify(8, 8, tiles, placements)
	found := false
	for _, v := range vs {
		if v.Kind == ViolationExceedsAtlas {
			found = true
			if got := v.BlockNumbers(); len(got) != 1 || got[0] != 1 {
				t.Fatalf("应报出块号 [1]，得到 %v", got)
			}
		}
	}
	if !found {
		t.Fatal("应检出图块超出图集")
	}
	if vs2 := CheckInput(tiles, 8); len(vs2) != 1 || vs2[0].Blocks[0] != 1 {
		t.Fatalf("CheckInput 应报出块号 1，得到 %v", vs2)
	}
}

func TestVerifyOverlapWithBlockNumbers(t *testing.T) {
	tiles := []*Gray{tile(4, 4, 1), tile(4, 4, 2), tile(2, 2, 3)}
	placements := []Placement{
		{Block: 0, X: 0, Y: 0, W: 4, H: 4},
		{Block: 1, X: 2, Y: 2, W: 4, H: 4},
		{Block: 2, X: 6, Y: 6, W: 2, H: 2},
	}
	vs := Verify(8, 8, tiles, placements)
	found := false
	for _, v := range vs {
		if v.Kind == ViolationOverlap {
			found = true
			if len(v.Blocks) != 2 || v.Blocks[0] != 0 || v.Blocks[1] != 1 {
				t.Fatalf("重叠应报出块号 [0 1]，得到 %v", v.Blocks)
			}
		}
	}
	if !found {
		t.Fatal("应检出图块重叠")
	}
}

func TestRenderRotatedMapping(t *testing.T) {
	g := &Gray{W: 3, H: 2, Pix: []uint8{0, 1, 2, 3, 4, 5}}
	atlas, err := render([]*Gray{g}, 5, []Placement{{Block: 0, W: 2, H: 3, Rotated: true}})
	if err != nil {
		t.Fatal(err)
	}
	// 源 3x2 顺时针 90° 后为 2x3:
	//   3 0
	//   4 1
	//   5 2
	if atlas.Pix[0] != 3 || atlas.Pix[1] != 0 ||
		atlas.Pix[5] != 4 || atlas.Pix[6] != 1 ||
		atlas.Pix[10] != 5 || atlas.Pix[11] != 2 {
		t.Fatalf("旋转映射错误: %v", atlas.Pix[:12])
	}
}

func TestPackIntoTileExceedsAtlasReportsBlock(t *testing.T) {
	tiles := []*Gray{tile(4, 4, 1), tile(9, 2, 2), tile(3, 3, 3)}
	_, err := PackInto(tiles, 8, Options{AllowRotation: true})
	if err == nil {
		t.Fatal("图块超出图集应报错")
	}
	v, ok := err.(*Violation)
	if !ok {
		t.Fatalf("错误类型应为 *Violation，得到 %T", err)
	}
	if v.Kind != ViolationExceedsAtlas {
		t.Fatalf("违规类别应为图块超出图集，得到 %v", v.Kind)
	}
	if got := v.BlockNumbers(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("应报出块号 [1]，得到 %v", got)
	}
}

func TestPackIntoRejectsSideBelowLowerBound(t *testing.T) {
	// 两块 5x5 单看都放得进 6x6，但面积下界为 ceil(sqrt(50))=8，故 6 已被下界否决。
	tiles := []*Gray{tile(5, 5, 0), tile(5, 5, 1)}
	if _, err := PackInto(tiles, 6, Options{AllowRotation: true}); !errors.Is(err, ErrNoFit) {
		t.Fatalf("边长低于下界应直接判不可行（ErrNoFit），得到 %v", err)
	}
	// 单块比图集还大时，按“图块超出图集”报块号，而不是笼统的不可行。
	if _, err := PackInto([]*Gray{tile(7, 3, 0)}, 6, Options{AllowRotation: true}); err == nil {
		t.Fatal("图块超出图集应报错")
	} else if v, ok := err.(*Violation); !ok || v.Kind != ViolationExceedsAtlas || v.BlockNumbers()[0] != 0 {
		t.Fatalf("应报图块超出图集且块号为 0，得到 %v", err)
	}
}

func TestPackIntoFixedSideSucceeds(t *testing.T) {
	tiles := []*Gray{tile(2, 2, 1), tile(3, 3, 2), tile(1, 4, 3)}
	res, err := PackInto(tiles, 6, Options{AllowRotation: true})
	if err != nil {
		t.Fatal(err)
	}
	checkResult(t, res, tiles)
	if res.Side != 6 {
		t.Fatalf("固定图集边长应为 6，得到 %d", res.Side)
	}
}

func TestPackRandomNoOverlap(t *testing.T) {
	for seed := 1; seed <= 20; seed++ {
		n := 3 + seed%6
		tiles := make([]*Gray, n)
		for i := range tiles {
			w := 1 + (seed*7+i*11)%23
			h := 1 + (seed*13+i*5)%19
			tiles[i] = tile(w, h, seed+i)
		}
		res := mustPack(t, tiles, Options{AllowRotation: seed%2 == 0})
		checkResult(t, res, tiles)
	}
}

func TestPackStressNoOverlap(t *testing.T) {
	for seed := 1; seed <= 5; seed++ {
		n := 120
		tiles := make([]*Gray, n)
		for i := range tiles {
			w := 1 + (seed*17+i*23)%60
			h := 1 + (seed*29+i*31)%55
			tiles[i] = tile(w, h, seed+i)
		}
		res := mustPack(t, tiles, Options{AllowRotation: true})
		checkResult(t, res, tiles)
	}
}

func TestPackEmptyTiles(t *testing.T) {
	if _, err := Pack(nil); err != ErrNoTiles {
		t.Fatalf("空列表应返回 ErrNoTiles，得到 %v", err)
	}
}
func TestPackWasteTable(t *testing.T) {
	cases := []struct {
		name  string
		tiles []*Gray
	}{
		{"等大", []*Gray{tile(32, 32, 1), tile(32, 32, 2), tile(32, 32, 3), tile(32, 32, 4)}},
		{"悬殊", []*Gray{tile(40, 40, 1), tile(12, 7, 2), tile(25, 3, 3), tile(5, 5, 4), tile(30, 12, 5)}},
		{"长条", []*Gray{tile(1, 100, 1), tile(100, 1, 2), tile(3, 80, 3)}},
		{"旋转有利", []*Gray{tile(100, 20, 1), tile(20, 100, 2)}},
		{"单块", []*Gray{tile(7, 3, 9)}},
	}
	t.Logf("%-8s %6s %6s %6s %10s %10s %10s %10s",
		"用例", "下界边", "边长", "上界边", "下界浪费", "实测浪费", "差值", "差上界")
	for _, c := range cases {
		res := mustPack(t, c.tiles, Options{AllowRotation: true})
		checkResult(t, res, c.tiles)
		t.Logf("%-8s %6d %6d %6d %10.6f %10.6f %10.6f %10.6f",
			c.name, res.LowerBoundSide, res.Side, res.UpperBoundSide,
			res.LowerBoundWaste, res.ActualWaste, res.Gap, res.GapUpperBound)
	}
}
