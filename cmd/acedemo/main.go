// acedemo 生成逆光/雾天/夜景/平坦/纯色五类用例,跑两档参数,
// 输出增强图与增益图 PGM,并打印截断统计。
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"goimage/img"
)

type lcg struct{ s uint32 }

func (l *lcg) next() uint32 {
	l.s = l.s*1664525 + 1013904223
	return l.s >> 24
}

func noise(g *img.Gray, amp int, seed uint32) {
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

func makeScene(name string) *img.Gray {
	g, _ := img.New(64, 48)
	switch name {
	case "backlit": // 逆光
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
	case "fog": // 雾天
		for y := 0; y < g.H; y++ {
			for x := 0; x < g.W; x++ {
				g.Pix[y*g.W+x] = uint8(150 + (x*20)/g.W)
			}
		}
		noise(g, 4, 2)
	case "night": // 夜景
		for i := range g.Pix {
			g.Pix[i] = 15
		}
		for _, p := range [][2]int{{10, 10}, {40, 20}, {30, 40}} {
			g.Pix[p[1]*g.W+p[0]] = 220
		}
		noise(g, 2, 3)
	case "flat": // 平坦
		for i := range g.Pix {
			g.Pix[i] = 128
		}
		noise(g, 1, 4)
	case "solid": // 纯色
		for i := range g.Pix {
			g.Pix[i] = 200
		}
	}
	return g
}

func main() {
	outDir := "artifacts"
	if len(os.Args) > 1 {
		outDir = os.Args[1]
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		panic(err)
	}

	tiers := []struct {
		name string
		p    img.ACE
	}{
		{"保守档", img.ACE{Radius: 64, TargetContrast: 0.10, MaxGain: 2.0}},
		{"增强档", img.ACE{Radius: 64, TargetContrast: 0.05, MaxGain: 4.0}},
	}
	scenes := []struct{ key, cn string }{
		{"backlit", "逆光"}, {"fog", "雾天"}, {"night", "夜景"},
		{"flat", "平坦"}, {"solid", "纯色"},
	}

	fmt.Printf("%-6s %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-6s\n",
		"场景", "档位", "反解半径", "实际半径", "过曝", "暗截断", "255档", "0档", "尖峰")
	for _, sc := range scenes {
		g := makeScene(sc.key)
		for _, tr := range tiers {
			res, err := img.EnhanceAdaptive(g, tr.p)
			if err != nil {
				fmt.Printf("%-6s %-6s 拒绝: %v\n", sc.cn, tr.name, err)
				continue
			}
			st := res.Stats
			spike := "无"
			if st.SpikeHigh || st.SpikeLow {
				spike = "有!"
			}
			fmt.Printf("%-6s %-6s %-8d %-8d %-8d %-8d %-8d %-8d %-6s\n",
				sc.cn, tr.name, st.RadiusSolved, st.RadiusUsed,
				st.Overexposed, st.ClippedLow, st.BinHigh, st.BinLow, spike)
			base := fmt.Sprintf("%s_%s", sc.key, map[string]string{"保守档": "mild", "增强档": "strong"}[tr.name])
			if err := res.Img.SavePGM(filepath.Join(outDir, base+"_out.pgm")); err != nil {
				panic(err)
			}
			if err := res.Gain.SavePGM(filepath.Join(outDir, base+"_gain.pgm")); err != nil {
				panic(err)
			}
		}
	}
}
