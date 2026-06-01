"""Document parsing tool. Extracts text from PDF, images, and plain text."""

from pathlib import Path
import os


class DocParser:
    SUPPORTED_FORMATS = {".pdf", ".txt", ".jpg", ".jpeg", ".png"}

    def parse(self, file_path: str) -> str:
        """Parse a file and return extracted text. Raises ValueError on failure."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的文件格式: {suffix}。"
                f"支持格式: " + ", ".join(sorted(self.SUPPORTED_FORMATS))
            )

        if suffix == ".pdf":
            return self._parse_pdf(file_path)
        elif suffix in {".jpg", ".jpeg", ".png"}:
            return self._parse_image(file_path)
        else:
            return self._parse_text(file_path)

    def _parse_pdf(self, file_path: str) -> str:
        try:
            import fitz
            doc = fitz.open(file_path)
            text = "\n".join(page.get_text() for page in doc)
            if not text.strip():
                ocr_text = self._ocr_pdf_pages(doc)
                doc.close()
                if ocr_text.strip():
                    return ocr_text.strip()
                raise ValueError("PDF \u6587\u4ef6\u4e2d\u672a\u68c0\u6d4b\u5230\u6587\u5b57\uff0c\u53ef\u80fd\u662f\u626b\u63cf\u4ef6\u56fe\u7247")
            doc.close()
            return text.strip()
        except ImportError:
            raise ImportError("\u8bf7\u5b89\u88c5 PyMuPDF: pip install pymupdf")
        except Exception as e:
            if "\u672a\u68c0\u6d4b\u5230\u6587\u5b57" in str(e):
                raise
            raise ValueError(f"PDF \u89e3\u6790\u5931\u8d25: {e}")

    def _ocr_pdf_pages(self, doc) -> str:
        """Try OCR on each page of a scanned/image-based PDF. Returns combined text."""
        from PIL import Image
        import io
        import tempfile

        texts = []
        for i, page in enumerate(doc):
            if i >= 10:
                break
            try:
                pix = page.get_pixmap(dpi=200)
            except Exception:
                continue
            img_data = pix.tobytes("png")

            page_text = ""
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_data)
                    tmp_path = tmp.name
                from src.tools.llm_tool import ocr_image
                page_text = ocr_image(tmp_path).strip()
            except Exception:
                pass
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

            if page_text:
                texts.append(page_text)
                continue

            try:
                import pytesseract
                tesseract_exe = os.environ.get("TESSERACT_CMD", "")
                if not tesseract_exe:
                    import platform
                    system = platform.system()
                    if system == "Windows":
                        candidates = [
                            "C:/Program Files/Tesseract-OCR/tesseract.exe",
                            "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
                        ]
                        for c in candidates:
                            if Path(c).exists():
                                tesseract_exe = c
                                break
                    elif system == "Darwin":
                        tesseract_exe = "/opt/homebrew/bin/tesseract"
                    else:
                        tesseract_exe = "/usr/bin/tesseract"
                if Path(tesseract_exe).exists():
                    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
                img = Image.open(io.BytesIO(img_data))
                img = self._preprocess_for_ocr(img)
                page_text = pytesseract.image_to_string(img, lang="chi_sim+eng", config="--psm 3").strip()
                if page_text:
                    texts.append(page_text)
            except Exception as e:
                print(f"[DocParser] PDF page OCR failed: {e}")

        return "\n".join(texts)

    def _preprocess_for_ocr(self, img):
        """Preprocess image to improve OCR accuracy for Chinese text."""
        from PIL import Image, ImageEnhance, ImageFilter
        if img.mode != "L":
            img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(1.5)
        w, h = img.size
        if w < 1200:
            scale = max(1.0, 1200.0 / w)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def _parse_image(self, file_path: str) -> str:
        try:
            from PIL import Image
            img = Image.open(file_path)
            img.load()
        except Exception:
            raise ValueError(f"图片文件无法打开或已损坏: {file_path}")

        # Try cloud vision OCR (Zhipu GLM-4V)
        try:
            from src.tools.llm_tool import ocr_image
            text = ocr_image(file_path)
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"[DocParser] Cloud OCR unavailable: {e}")

        # Fall back to local Tesseract OCR
        try:
            import pytesseract
            from PIL import Image
            import platform
            tesseract_exe = os.environ.get("TESSERACT_CMD", "")
            if not tesseract_exe:
                system = platform.system()
                if system == "Windows":
                    candidates = [
                        "C:/Program Files/Tesseract-OCR/tesseract.exe",
                        "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
                    ]
                    for c in candidates:
                        if Path(c).exists():
                            tesseract_exe = c
                            break
                elif system == "Darwin":
                    tesseract_exe = "/opt/homebrew/bin/tesseract"
                else:
                    tesseract_exe = "/usr/bin/tesseract"
            if Path(tesseract_exe).exists():
                pytesseract.pytesseract.tesseract_cmd = tesseract_exe
            else:
                raise RuntimeError("Tesseract binary not found")
            img = Image.open(file_path)
            img = self._preprocess_for_ocr(img)
            text = pytesseract.image_to_string(img, lang="chi_sim+eng", config="--psm 3")
            if text.strip():
                return text.strip()
        except ImportError:
            print("[DocParser] pytesseract not installed, Tesseract OCR unavailable")
        except Exception as e:
            print(f"[DocParser] Tesseract OCR failed: {e}")

        raise ValueError(
            "图片文字识别失败。\u8bf7确认已配置智谱 API Key\uff08\u63a8荐\uff09\u6216安装 Tesseract-OCR\uff0c"
            "或上传 PDF/\u6587本格式的\u6587件。"
        )

    def _parse_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError("\u6587件内容为空")
        return text

    def parse_or_empty(self, file_path: str) -> str:
        """Parse file, return empty string on failure instead of raising."""
        try:
            return self.parse(file_path)
        except Exception as e:
            print(f"[DocParser] 解析失败 {file_path}: {e}")
            return ""
