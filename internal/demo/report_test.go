package demo

import (
	"bytes"
	"strings"
	"testing"

	"goimage/img"
)

func TestReportMonotonicAndDCLossless(t *testing.T) {
	var buf bytes.Buffer
	ReportTo(&buf, []int{8, 32})
	out := buf.String()
	if n := strings.Count(out, "无损"); n < 10 {
		t.Fatalf("每例两档都应标为直流无损，只找到 %d 处:\n%s", n, out)
	}
	if strings.Contains(out, "有损") || strings.Contains(out, "↓") {
		t.Fatalf("报告出现违反约束的行:\n%s", out)
	}
	if strings.Contains(out, "失败") {
		t.Fatalf("报告包含失败:\n%s", out)
	}
}

func TestFixturesDistinct(t *testing.T) {
	seen := map[uint64]string{}
	for _, c := range Cases() {
		g := c.New()
		if len(g.Pix) != Size*Size {
			t.Fatalf("%s 尺寸错误", c.Name)
		}
		var h uint64 = 1469598103934665603
		for _, v := range g.Pix {
			h = (h ^ uint64(v)) * 1099511628211
		}
		if prev, ok := seen[h]; ok {
			t.Fatalf("%s 与 %s 内容相同", c.Name, prev)
		}
		seen[h] = c.Name
	}
}

func TestMetrics(t *testing.T) {
	a, _ := img.New(2, 2)
	b, _ := img.New(2, 2)
	a.Pix = []uint8{0, 100, 200, 255}
	b.Pix = []uint8{10, 90, 210, 255}
	mse := MSE(a, b)
	if mse != (100+100+100)/4.0 {
		t.Fatalf("MSE 计算错误: %v", mse)
	}
	if MaxErr(a, b) != 10 {
		t.Fatalf("最大误差错误: %d", MaxErr(a, b))
	}
	if got := PSNR(0); !isInf(got) {
		t.Fatalf("零误差 PSNR 应为 +Inf，实为 %v", got)
	}
}

func isInf(v float64) bool { return v > 1e300 }
