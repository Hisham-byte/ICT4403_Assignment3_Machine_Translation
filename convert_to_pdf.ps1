$baseDir = "C:\Users\Admin\Documents\New folder\ICT4403_Assignment3_Machine_Translation"

$files = @(
    "HishamAdil_230953434_FGAI_Assignment3",
    "YourName_RegNum_FGAI_Assignment3"
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    foreach ($f in $files) {
        $docPath = [string](Join-Path $baseDir "$f.docx")
        $pdfPath = [string](Join-Path $baseDir "$f.pdf")
        if (Test-Path $docPath) {
            $doc = $word.Documents.Open($docPath)
            $doc.SaveAs([ref]$pdfPath, [ref]17)
            $doc.Close()
            Write-Output "Successfully exported PDF: $pdfPath"
        }
    }
} catch {
    Write-Error "Conversion error: $_"
} finally {
    $word.Quit()
}
