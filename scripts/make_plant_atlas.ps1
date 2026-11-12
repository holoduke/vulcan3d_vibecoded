# Builds assets/foliage/plant_atlas.png (512x256): a leaf-surface patch
# (left) and a petal-surface patch (right, near-white so it can be tinted)
# for the small plants (flower leaves, fern leaflets, petals).
# Source: the largest fully-covered square inside a leaf of the oak twig
# photo (assets/foliage/src/leaf.png, magenta background).
Add-Type -AssemblyName System.Drawing
$root = Split-Path -Parent $PSScriptRoot
$src = [Drawing.Bitmap]::FromFile("$root\assets\foliage\src\leaf.png")
$W = $src.Width; $H = $src.Height
$d = $src.LockBits((New-Object Drawing.Rectangle 0, 0, $W, $H), 'ReadOnly', ([Drawing.Imaging.PixelFormat]::Format32bppArgb))
$buf = New-Object byte[] ($d.Stride * $H)
[Runtime.InteropServices.Marshal]::Copy($d.Scan0, $buf, 0, $buf.Length)
$src.UnlockBits($d)
# Leaf mask on a 4-px grid: green dominant, not magenta, not the dark twig.
$S = 4; $gw = [int]($W / $S); $gh = [int]($H / $S)
$dp = New-Object 'int[,]' $gh, $gw
$best = 0; $bx = 0; $by = 0
for ($y = 0; $y -lt $gh; $y++) {
    for ($x = 0; $x -lt $gw; $x++) {
        $o = ($y * $S) * $d.Stride + ($x * $S) * 4
        $b = [int]$buf[$o]; $g = [int]$buf[$o + 1]; $r = [int]$buf[$o + 2]
        $leaf = ($g -gt $r + 8) -and ($g -gt $b + 8) -and ($g -gt 50)
        if ($leaf) {
            $v = 1
            if ($x -gt 0 -and $y -gt 0) {
                $ym = $y - 1; $xm = $x - 1
                $a1 = $dp[$ym, $x]; $a2 = $dp[$y, $xm]; $a3 = $dp[$ym, $xm]
                $v = 1 + [Math]::Min([Math]::Min($a1, $a2), $a3)
            }
            $dp[$y, $x] = $v
            if ($v -gt $best) { $best = $v; $bx = $x; $by = $y }
        } else { $dp[$y, $x] = 0 }
    }
}
$side = $best * $S; $x0 = ($bx - $best + 1) * $S; $y0 = ($by - $best + 1) * $S
"largest leaf square: $side px at ($x0, $y0)"
$atlas = New-Object Drawing.Bitmap 512, 256, ([Drawing.Imaging.PixelFormat]::Format24bppRgb)
$g = [Drawing.Graphics]::FromImage($atlas); $g.InterpolationMode = 'HighQualityBicubic'
$rect = New-Object Drawing.Rectangle $x0, $y0, $side, $side
$g.DrawImage($src, (New-Object Drawing.Rectangle 0, 0, 256, 256), $rect, 'Pixel')
# Petal: the same structure, desaturated and lifted to near white.
$cm = New-Object Drawing.Imaging.ColorMatrix
$cm.Matrix00 = 0.45; $cm.Matrix01 = 0.45; $cm.Matrix02 = 0.45
$cm.Matrix10 = 0.88; $cm.Matrix11 = 0.88; $cm.Matrix12 = 0.88
$cm.Matrix20 = 0.17; $cm.Matrix21 = 0.17; $cm.Matrix22 = 0.17
$cm.Matrix40 = 0.30; $cm.Matrix41 = 0.30; $cm.Matrix42 = 0.30
$ia = New-Object Drawing.Imaging.ImageAttributes; $ia.SetColorMatrix($cm)
$g.DrawImage($src, (New-Object Drawing.Rectangle 256, 0, 256, 256), $x0, $y0, $side, $side, 'Pixel', $ia)
$g.Dispose(); $src.Dispose()
$atlas.Save("$root\assets\foliage\plant_atlas.png", [Drawing.Imaging.ImageFormat]::Png)
$atlas.Dispose()
