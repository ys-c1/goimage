package img

import "testing"

// 逐项核对：内置核的每一项偏移、权重都必须与标准定义一致。
// 缺一项、多一项、顺序或数值不同都算失败。
func TestBuiltinKernelWeightTables(t *testing.T) {
	type tap struct {
		dx, dy int
		w      float64
	}
	specs := map[string][]tap{
		"floyd-steinberg": {
			{1, 0, 7.0 / 16}, {-1, 1, 3.0 / 16}, {0, 1, 5.0 / 16}, {1, 1, 1.0 / 16},
		},
		"jarvis-judice-ninke": {
			{1, 0, 7.0 / 48}, {2, 0, 5.0 / 48},
			{-2, 1, 3.0 / 48}, {-1, 1, 5.0 / 48}, {0, 1, 7.0 / 48}, {1, 1, 5.0 / 48}, {2, 1, 3.0 / 48},
			{-2, 2, 1.0 / 48}, {-1, 2, 3.0 / 48}, {0, 2, 5.0 / 48}, {1, 2, 3.0 / 48}, {2, 2, 1.0 / 48},
		},
		"stucki": {
			{1, 0, 8.0 / 42}, {2, 0, 4.0 / 42},
			{-2, 1, 2.0 / 42}, {-1, 1, 4.0 / 42}, {0, 1, 8.0 / 42}, {1, 1, 4.0 / 42}, {2, 1, 2.0 / 42},
			{-2, 2, 1.0 / 42}, {-1, 2, 2.0 / 42}, {0, 2, 4.0 / 42}, {1, 2, 2.0 / 42}, {2, 2, 1.0 / 42},
		},
		"atkinson": {
			{1, 0, 1.0 / 8}, {2, 0, 1.0 / 8},
			{-1, 1, 1.0 / 8}, {0, 1, 1.0 / 8}, {1, 1, 1.0 / 8},
			{0, 2, 1.0 / 8},
		},
		"burkes": {
			{1, 0, 8.0 / 32}, {2, 0, 4.0 / 32},
			{-2, 1, 2.0 / 32}, {-1, 1, 4.0 / 32}, {0, 1, 8.0 / 32}, {1, 1, 4.0 / 32}, {2, 1, 2.0 / 32},
		},
		"sierra": {
			{1, 0, 5.0 / 32}, {2, 0, 3.0 / 32},
			{-2, 1, 2.0 / 32}, {-1, 1, 4.0 / 32}, {0, 1, 5.0 / 32}, {1, 1, 4.0 / 32}, {2, 1, 2.0 / 32},
			{-1, 2, 2.0 / 32}, {0, 2, 3.0 / 32}, {1, 2, 2.0 / 32},
		},
		"sierra-lite": {
			{1, 0, 2.0 / 4}, {-1, 1, 1.0 / 4}, {0, 1, 1.0 / 4},
		},
	}

	builtins := BuiltinKernels()
	if len(builtins) != len(specs) {
		t.Fatalf("内置核数量 %d，期望 %d", len(builtins), len(specs))
	}
	seen := map[string]bool{}
	for _, k := range builtins {
		if seen[k.Name] {
			t.Fatalf("核 %q 重复", k.Name)
		}
		seen[k.Name] = true
		want, ok := specs[k.Name]
		if !ok {
			t.Fatalf("未登记的内置核 %q", k.Name)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("%s 自身校验失败：%v", k.Name, err)
		}
		if len(k.DX) != len(want) || len(k.DY) != len(want) || len(k.W) != len(want) {
			t.Fatalf("%s 项数 %d/%d/%d，期望 %d", k.Name, len(k.DX), len(k.DY), len(k.W), len(want))
		}
		for i, w := range want {
			if k.DX[i] != w.dx || k.DY[i] != w.dy || k.W[i] != w.w {
				t.Fatalf("%s 第 %d 项 = (%d,%d,%v)，期望 (%d,%d,%v)",
					k.Name, i, k.DX[i], k.DY[i], k.W[i], w.dx, w.dy, w.w)
			}
		}
	}
	for name := range specs {
		if !seen[name] {
			t.Fatalf("缺少内置核 %q", name)
		}
	}
	// 返回副本：改动不应影响内置表。
	got, _ := LookupKernel("floyd-steinberg")
	got.W[0] = 0
	again, _ := LookupKernel("floyd-steinberg")
	if again.W[0] != 7.0/16 {
		t.Fatal("BuiltinKernels/LookupKernel 未返回副本")
	}
	// 名字归一化：大小写与下划线都接受。
	if k, err := LookupKernel("Sierra_Lite"); err != nil || k.Name != "sierra-lite" {
		t.Fatalf("名字归一化失败：%v", err)
	}
}
