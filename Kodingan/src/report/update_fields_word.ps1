# Isi field daftar isi, daftar tabel, dan daftar gambar dengan nomor halaman
# sungguhan, lalu simpan ulang .docx dan ekspor PDF-nya.
#
# python-docx hanya dapat menulis kode field, bukan menghitung tata letak halaman,
# sehingga pengisian nomor halaman harus dikerjakan Word sendiri.
#
# Pemakaian: powershell -File update_fields_word.ps1 <masukan.docx> <keluaran.pdf>
param([string]$Docx, [string]$Pdf)

$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::GetFullPath($Pdf)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false)
    # Dua kali: pembaruan pertama dapat menggeser halaman karena daftar isi
    # bertambah panjang, pembaruan kedua mengoreksi nomor halamannya.
    for ($k = 0; $k -lt 2; $k++) {
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        $doc.Fields.Update() | Out-Null
        $doc.Repaginate()
    }
    $doc.Save()
    $doc.SaveAs([ref]$Pdf, [ref]17)
    $doc.Close([ref]0)
    "ok: $Docx"
    "ok: $Pdf"
}
finally {
    # Hanya tutup Word bila tidak ada dokumen lain yang terbuka di instans ini,
    # agar dokumen yang sedang dikerjakan pengguna tidak ikut tertutup.
    if ($word.Documents.Count -eq 0) { $word.Quit() }
}
