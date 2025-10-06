import io, os
from typing import Iterable
from pypdf import PdfReader

def read_pdfs(files: Iterable) -> str:
    """Read uploaded PDFs and return a single text string."""
    texts = []
    for f in files:
        try:
            if hasattr(f, 'read') and callable(f.read):
                content = f.read()
            elif hasattr(f, 'file') and hasattr(f.file, 'read') and callable(f.file.read):
                content = f.file.read()
            elif hasattr(f, 'name') and isinstance(f.name, str) and os.path.isfile(f.name):
                with open(f.name, 'rb') as file_obj:
                    content = file_obj.read()
            else:
                content = (getattr(f, 'data', None) or getattr(f, 'content', None) or str(f)).encode('utf-8')
            pdf = PdfReader(io.BytesIO(content))
            pages = []
            for page in pdf.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            texts.append("\n".join(pages))
        except Exception as e:
            print(f"⚠️ Error reading {getattr(f, 'name', str(f))}: {e}")
            texts.append("")
    return "\n\n".join(texts)
