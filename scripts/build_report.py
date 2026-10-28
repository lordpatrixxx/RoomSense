"""Create a self-contained printable HTML report with executed results."""
import base64
from pathlib import Path
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]


def build():
    report = (ROOT / "reports/PROJECT_REPORT.md").read_text(encoding="utf-8")
    results = (ROOT / "reports/RESULTS.md").read_text(encoding="utf-8")
    report = report.replace("## Experimental protocol and results", "## Experimental protocol and results\n\n" + results.replace("# Reproduced experimental results", "### Executed results").replace("## Diagnostic test comparisons", "### Diagnostic test comparisons"))
    body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    body = body.replace('See <a href="RESULTS.md">Reproduced experimental results</a>', 'See the executed results above')
    def embed(match):
        file = ROOT / "reports" / match.group(1)
        if not file.exists():
            return match.group(0)
        return 'src="data:image/png;base64,' + base64.b64encode(file.read_bytes()).decode() + '"'
    body = re.sub(r'src="(figures/[^\"]+)"', embed, body)
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomSense — Mini project report</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;color:#193344;background:#edf2f4;margin:0;line-height:1.6}
main{max-width:920px;margin:32px auto;padding:48px 64px;background:white;border-radius:12px}
h1{font-size:32px;line-height:1.2;color:#075c55;margin:0 0 28px}h2{font-size:23px;margin-top:36px}h3{font-size:18px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th{background:#eaf2f3;text-align:left}
th,td{padding:9px 12px;border-bottom:1px solid #dce5e9;vertical-align:top}
img{max-width:100%;height:auto;display:block;margin:24px auto}a{color:#007f73;overflow-wrap:anywhere}
code{background:#f0f4f6;padding:2px 4px;overflow-wrap:anywhere}li{margin:6px 0}p{margin:12px 0}
.print-note{font-size:13px;color:#48606f;background:#f1f7f6;padding:12px 16px;margin-bottom:24px}
@media print{body{background:white;font-size:10pt}main{max-width:none;margin:0;padding:0;border:0}
@page{size:A4;margin:18mm}h1{font-size:23pt}h2{font-size:16pt;break-after:avoid}h3{break-after:avoid}
table{font-size:9pt}tr,img{break-inside:avoid}.print-note{display:none}a{color:inherit}p{orphans:3;widows:3}}
</style></head><body><main><div class="print-note">RoomSense project report · Includes executed results · Use your browser's Print command to save as PDF.</div>""" + body + "</main></body></html>"
    target = ROOT / "reports/PROJECT_REPORT.html"
    target.write_text(html, encoding="utf-8")
    print(f"Printable report saved: {target}")


if __name__ == "__main__":
    build()
