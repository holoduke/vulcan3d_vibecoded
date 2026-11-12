# Builds assets/foliage/leaf_atlas.png (2048x1024 RGBA: broadleaf twig left,
# conifer spray right) from the magenta-keyed source images, and
# assets/foliage/bark.png.
Add-Type -AssemblyName System.Drawing
$root = 'C:\Users\gillis\dev\quake-like\assets\foliage'
$S = 1024
$atlas = New-Object Drawing.Bitmap (2 * $S), $S, ([Drawing.Imaging.PixelFormat]::Format32bppArgb)
$ad = $atlas.LockBits((New-Object Drawing.Rectangle 0, 0, (2 * $S), $S), 'WriteOnly', $atlas.PixelFormat)
$out = New-Object byte[] ($ad.Stride * $S)
$i = 0
foreach ($name in 'leaf', 'needle') {
    $src0 = [Drawing.Bitmap]::FromFile("$root\src\$name.png")
    $src = New-Object Drawing.Bitmap $S, $S, ([Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [Drawing.Graphics]::FromImage($src); $g.InterpolationMode = 'HighQualityBicubic'
    $g.DrawImage($src0, 0, 0, $S, $S); $g.Dispose(); $src0.Dispose()
    $sd = $src.LockBits((New-Object Drawing.Rectangle 0, 0, $S, $S), 'ReadOnly', $src.PixelFormat)
    $buf = New-Object byte[] ($sd.Stride * $S)
    [Runtime.InteropServices.Marshal]::Copy($sd.Scan0, $buf, 0, $buf.Length)
    $src.UnlockBits($sd); $src.Dispose()
    # Fill colour for the transparent area (stops a magenta fringe in the mips).
    $fill = if ($name -eq 'leaf') { @(30, 110, 50) } else { @(30, 70, 30) }   # B, G, R
    for ($y = 0; $y -lt $S; $y++) {
        for ($x = 0; $x -lt $S; $x++) {
            $o = $y * $sd.Stride + $x * 4
            $b = [int]$buf[$o]; $gr = [int]$buf[$o + 1]; $r = [int]$buf[$o + 2]
            $m = [Math]::Min($r, $b) - $gr              # magenta-ness
            $a = 1.0 - [Math]::Min(1.0, [Math]::Max(0.0, ($m - 25.0) / 70.0))
            $p = $y * $ad.Stride + ($i * $S + $x) * 4
            if ($a -le 0.02) {
                $out[$p] = $fill[0]; $out[$p + 1] = $fill[1]; $out[$p + 2] = $fill[2]; $out[$p + 3] = 0
            } else {
                # De-spill: magenta raises red and blue above green.
                $r2 = [Math]::Min($r, [int]($gr * 1.25) + 12); $b2 = [Math]::Min($b, [int]($gr * 0.85) + 6)
                if ($a -lt 0.999) { $r2 = [int]($r2 * $a + $fill[2] * (1 - $a)); $b2 = [int]($b2 * $a + $fill[0] * (1 - $a)) }
                $out[$p] = [byte]$b2; $out[$p + 1] = [byte]$gr; $out[$p + 2] = [byte]$r2; $out[$p + 3] = [byte]([int]($a * 255))
            }
        }
    }
    $i++
}
[Runtime.InteropServices.Marshal]::Copy($out, 0, $ad.Scan0, $out.Length)
$atlas.UnlockBits($ad)
$atlas.Save("$root\leaf_atlas.png", [Drawing.Imaging.ImageFormat]::Png)
$atlas.Dispose()
$b0 = [Drawing.Bitmap]::FromFile("$root\src\bark.png")
$bk = New-Object Drawing.Bitmap 1024, 1024, ([Drawing.Imaging.PixelFormat]::Format24bppRgb)
$g = [Drawing.Graphics]::FromImage($bk); $g.InterpolationMode = 'HighQualityBicubic'; $g.DrawImage($b0, 0, 0, 1024, 1024); $g.Dispose(); $b0.Dispose()
$bk.Save("$root\bark.png", [Drawing.Imaging.ImageFormat]::Png); $bk.Dispose()
Get-ChildItem $root -File | ForEach-Object { "$($_.Name) $($_.Length)" }
