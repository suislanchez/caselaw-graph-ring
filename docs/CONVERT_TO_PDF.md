# How to Convert Research Context to PDF

## Option 1: Online Converter (Easiest)
1. Go to https://md2pdf.netlify.app/ or https://www.markdowntopdf.com/
2. Copy contents of `RESEARCH_CONTEXT.md`
3. Paste and download PDF

## Option 2: VS Code Extension
1. Install "Markdown PDF" extension in VS Code
2. Open `RESEARCH_CONTEXT.md`
3. Cmd+Shift+P → "Markdown PDF: Export (pdf)"

## Option 3: Install Pandoc (Best Quality)
```bash
brew install pandoc
brew install --cask basictex  # For PDF support

# Convert to PDF
pandoc RESEARCH_CONTEXT.md -o RESEARCH_CONTEXT.pdf \
  --pdf-engine=pdflatex \
  -V geometry:margin=1in \
  -V fontsize=11pt

# Or convert to HTML first (no LaTeX needed)
pandoc RESEARCH_CONTEXT.md -o RESEARCH_CONTEXT.html --standalone
```

## Option 4: Python Script
```bash
pip install markdown weasyprint
python -c "
import markdown
from weasyprint import HTML

with open('RESEARCH_CONTEXT.md') as f:
    html = markdown.markdown(f.read(), extensions=['tables', 'fenced_code'])

HTML(string=f'<html><body>{html}</body></html>').write_pdf('RESEARCH_CONTEXT.pdf')
"
```

## Option 5: macOS Preview
1. Open `RESEARCH_CONTEXT.md` in any Markdown viewer
2. File → Print → Save as PDF
