package demo

import (
	"fmt"
	"io"
	"os"

	"goimage/img"
)

// Steps 是报告使用的两档量化步长。
var Steps = []int{8, 32}

// Report 跑全部用例、两档步长，打印误差曲线与压缩比，并逐项核对
// 两条硬性要求：误差随步长单调上升、直流无损。
func Report() {
	ReportTo(os.Stdout, Steps)
}

// ReportTo 便于测试捕获输出。
func ReportTo(w io.Writer, steps []int) {
	tbl := img.HVSTable()
	fmt.Fprintf(w, "量化表（由人眼对比敏感度推导，直流不量化，权重为 2 的幂）:\n")
	for row := 0; row < img.BlockSize; row++ {
		for col := 0; col < img.BlockSize; col++ {
			fmt.Fprintf(w, "%3d", tbl[row][col])
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "\n块尺寸 %d，步长档位:", img.BlockSize)
	for _, s := range steps {
		fmt.Fprintf(w, " %d", s)
	}
	fmt.Fprintln(w)

	fmt.Fprintf(w, "\n%-6s %6s %9s %9s %7s %6s %8s  %s\n",
		"用例", "步长", "MSE", "PSNR(dB)", "最大误差", "系数位", "压缩比", "直流")
	var prevMSE = map[string]float64{}
	var prevStep = map[string]int{}
	for _, c := range Cases() {
		g := c.New()
		for _, step := range steps {
			d, err := img.EncodeDCT(g, img.BlockSize, step)
			if err != nil {
				fmt.Fprintf(w, "%-6s 编码失败: %v\n", c.Name, err)
				continue
			}
			stream, err := d.EncodeStream()
			if err != nil {
				fmt.Fprintf(w, "%-6s 打包失败: %v\n", c.Name, err)
				continue
			}
			back, err := img.DecodeStream(stream, g.W, g.H, step)
			if err != nil {
				fmt.Fprintf(w, "%-6s 解包失败: %v\n", c.Name, err)
				continue
			}
			out, err := back.Decode()
			if err != nil {
				fmt.Fprintf(w, "%-6s 重建失败: %v\n", c.Name, err)
				continue
			}
			mse := MSE(g, out)
			dcOK := pixelSum(g) == pixelSum(out)
			trend := "—"
			if p, ok := prevMSE[c.Name]; ok {
				switch {
				case mse > p:
					trend = "↑"
				case mse == p:
					trend = "持平"
				default:
					trend = fmt.Sprintf("↓(违规，前一步长 %d)", prevStep[c.Name])
				}
			}
			prevMSE[c.Name] = mse
			prevStep[c.Name] = step
			dc := "无损"
			if !dcOK {
				dc = "有损"
			}
			fmt.Fprintf(w, "%-6s %6d %9.4f %9.3f %7d %7d %7.2f:1  %s %s\n",
				c.Name, step, mse, PSNR(mse), MaxErr(g, out), d.EncodedBits(), d.Ratio(), dc, trend)
		}
	}

	fmt.Fprintln(w, "\n说明:")
	fmt.Fprintln(w, "  误差曲线：每档步长各一行；末列是相对更小步长的走向，只允许 ↑ 或持平。")
	fmt.Fprintln(w, "  直流：存储的 DC 是原块像素和，重建时块和逐位回填，因此与步长无关。")
	fmt.Fprintln(w, "  纯色没有 AC 分量，任何步长都应精确重建，故误差恒为 0（持平）。")
	fmt.Fprintln(w, "  压缩比：原始 8bpp 位数 ÷ 实际打包的系数流位数（含块头与 EOB）。")
}

func pixelSum(g *img.Gray) int {
	s := 0
	for _, v := range g.Pix {
		s += int(v)
	}
	return s
}
