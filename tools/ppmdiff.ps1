param(
    [Parameter(Mandatory=$true)][string]$Base,   # e.g. burst2  (expects Base.NN.ppm)
    [int]$Count = 8,
    [string]$OutPrefix = "diff",
    [int]$Amp = 6
)
$cs = @'
using System;
using System.IO;
public static class PpmDiff {
    public static byte[] Load(string path, out int w, out int h) {
        byte[] f = File.ReadAllBytes(path);
        int i = 0; int fields = 0; string tok = ""; w = 0; h = 0;
        while (fields < 4) {
            char c = (char)f[i];
            if (char.IsWhiteSpace(c)) { if (tok.Length > 0) { fields++; if (fields == 2) w = int.Parse(tok); if (fields == 3) h = int.Parse(tok); tok = ""; } }
            else tok += c;
            i++;
        }
        byte[] px = new byte[w * h * 3];
        Array.Copy(f, i, px, 0, px.Length);
        return px;
    }
    public static string Stats(string a, string b, string outPng, int amp) {
        int w, h, w2, h2;
        byte[] pa = Load(a, out w, out h);
        byte[] pb = Load(b, out w2, out h2);
        long sum = 0; int over8 = 0, over20 = 0; int maxd = 0;
        byte[] diff = new byte[w * h * 3];
        for (int k = 0; k < w * h; k++) {
            int d = 0;
            for (int c = 0; c < 3; c++) {
                int dd = Math.Abs(pa[k*3+c] - pb[k*3+c]);
                if (dd > d) d = dd;
            }
            sum += d;
            if (d > 8) over8++;
            if (d > 20) over20++;
            if (d > maxd) maxd = d;
            int v = Math.Min(255, d * amp);
            diff[k*3] = (byte)v; diff[k*3+1] = (byte)v; diff[k*3+2] = (byte)v;
        }
        using (var bmp = new System.Drawing.Bitmap(w, h, System.Drawing.Imaging.PixelFormat.Format24bppRgb)) {
            var bd = bmp.LockBits(new System.Drawing.Rectangle(0,0,w,h), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            for (int y = 0; y < h; y++) {
                byte[] row = new byte[bd.Stride];
                for (int x = 0; x < w; x++) { row[x*3] = diff[(y*w+x)*3]; row[x*3+1] = diff[(y*w+x)*3+1]; row[x*3+2] = diff[(y*w+x)*3+2]; }
                System.Runtime.InteropServices.Marshal.Copy(row, 0, (IntPtr)(bd.Scan0.ToInt64() + y * bd.Stride), bd.Stride);
            }
            bmp.UnlockBits(bd);
            bmp.Save(outPng, System.Drawing.Imaging.ImageFormat.Png);
        }
        return string.Format("mean={0:F2} px>8={1} px>20={2} max={3}", (double)sum/(w*h), over8, over20, maxd);
    }
}
'@
Add-Type -TypeDefinition $cs -ReferencedAssemblies System.Drawing
for ($i = 0; $i -lt $Count - 1; $i++) {
    $a = "{0}.{1:D2}.ppm" -f $Base, $i
    $b = "{0}.{1:D2}.ppm" -f $Base, ($i + 1)
    $o = "{0}_{1}.png" -f $OutPrefix, $i
    $s = [PpmDiff]::Stats($a, $b, $o, $Amp)
    "pair ${i}->$($i+1): $s"
}
