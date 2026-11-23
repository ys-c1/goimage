package img

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	g, err := New(4, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Pix {
		g.Pix[i] = uint8(i * 5)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "a.pgm")
	if err := g.SavePGM(p); err != nil {
		t.Fatal(err)
	}
	got, err := LoadPGM(p)
	if err != nil {
		t.Fatal(err)
	}
	if got.W != 4 || got.H != 3 || len(got.Pix) != 12 {
		t.Fatalf("尺寸错误 %d %d", got.W, got.H)
	}
	for i := range got.Pix {
		if got.Pix[i] != g.Pix[i] {
			t.Fatalf("第 %d 像素不一致", i)
		}
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatal("文件未生成")
	}
}

func TestQuantizeLevels(t *testing.T) {
	g, _ := New(4, 1)
	g.Pix = []uint8{0, 60, 130, 250}
	out, err := Quantize(g, 4)
	if err != nil {
		t.Fatal(err)
	}
	if LevelsUsed(out) > 4 {
		t.Fatalf("色阶超限 %d", LevelsUsed(out))
	}
	if _, err := Quantize(g, 0); err == nil {
		t.Fatal("色阶为零应报错")
	}
}

func TestNewRejectsZero(t *testing.T) {
	if _, err := New(0, 2); err == nil {
		t.Fatal("宽度为零应报错")
	}
}
