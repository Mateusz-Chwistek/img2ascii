import requests
import numpy as np
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from PIL.Image import DecompressionBombError
from PIL import Image, ImageFont, ImageDraw, UnidentifiedImageError

class ImgConverterError(Exception):
    """Base ImgConverter exception"""
    pass

class ImageNotLoadedError(ImgConverterError):
    """Raised when an image is not loaded properly"""
    pass

class InvalidURLError(ImgConverterError):
    """Raised when a provided URL is invalid"""
    pass

class UnsupportedExtensionError(ImgConverterError):
    """Raised when the file extension is not supported"""
    pass

class ImageDownloadError(ImgConverterError):
    """Raised when an image cannot be downloaded"""
    pass

class TargetExistsError(ImgConverterError):
    """Raised when a file already exists at the target location"""
    pass

class ImgConverter:
    _SUPPORTED_FORMATS = ("jpeg", "jpg", "png", "webp",
                      "tif", "tiff", "bmp", "ppm",
                      "pgm", "pbm", "ico")
    _CHARSETS = {
        "simple": "@%#*+=-:. ",
        "advanced": "@$B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvu "
                    "nxrjft/\\|()1}{][?-_+~i!lI;:,\"^`'. "
    }
    _MAX_SCALE = 6
    _PIXELS_LIMIT = 8_847_360 # DCI-4K 4096x2160
    _BYTES_PER_PIXEL_WORST = 8 # 4 channels * 2 bytes
    _IMAGE_SIZE_LIMIT = _PIXELS_LIMIT * _BYTES_PER_PIXEL_WORST
    Image.MAX_IMAGE_PIXELS = _PIXELS_LIMIT
    
    def __init__(self):
        """Initializes the ImgConverter instance."""
        self._img_array = self._ascii_array = self._font = self._scale = self._advanced_set = None
    
    def _convert_to_grayscale(self, img: Image.Image):
        """
        Extracts green channel, and zero-out fully transparent pixels.
        
        Args:
            img (PIL.Image.Image): Input pillow image.
        
        Returns:
            np.ndarray: 2D array (H x W) of green-channel values, dtype uint8.
        
        Raises:
            ImgConverterError: If `img` is not a pillow Image.
        """
        if not isinstance(img, Image.Image):
            raise ImgConverterError("Image is not a pillow image")
        
        img_rgba = img.convert("RGBA")

        rgba = np.asarray(img_rgba, dtype=np.uint8)

        if rgba.ndim != 3 or rgba.shape[-1] != 4:
            raise ValueError("Expected RGBA array of shape (H, W, 4)")

        luma = np.dot(rgba[..., :3].astype(np.float32), [0.2126, 0.7152, 0.0722])

        alpha = rgba[..., 3]
        luma = np.where(alpha == 0, 255, luma)

        return luma.astype(np.uint8)
    
    def _convert_to_ascii(self, scale: int, advanced_set: bool, skip_rows: bool) -> None:
        """
        Converts the loaded image to an ASCII representation.
        
        Args:
            scale (int): Power of 2 that defines the size of the block window used for downscaling.
            advanced_set (bool): If True, uses a more detailed character set.
            skip_rows (bool): If True, uses every second row of pixels, for better display in console.
        
        Raises:
            ImageNotLoadedError: If no image is loaded.
        """
        if not isinstance(advanced_set, bool):
            raise ValueError("advanced_set must be a boolean")
        
        if not isinstance(scale, int):
            raise ValueError("scale must be a integer")
        
        charset = self._CHARSETS["advanced"] if advanced_set else self._CHARSETS["simple"]
        chars = np.asarray(list(charset), dtype="<U1")

        if self._img_array is None:
            raise ImageNotLoadedError("Load image first")

        if skip_rows:
            arr = self._img_array[::2]
        else:
            arr = self._img_array
        
        h, w = arr.shape
        clamped_scale = max(0, min(scale, self._MAX_SCALE))
        block = 1 << clamped_scale
        h -= h % block
        w -= w % block
        arr = arr[:h, :w]

        if block > 1:
            arr = arr.reshape(h // block, block, w // block, block).mean((1, 3)).astype(np.uint8)

        self._ascii_array = chars[(arr.astype(np.uint16) * (len(chars)-1) // 255)]
        self._scale = scale
        self._advanced_set = advanced_set
    
    def _is_supported_format(self, format: str | None) -> bool:
        """
        Check if the file has a supported format.

        Args:
            format (str | None): File format received from Image.format

        Returns:
            bool: True if the file has a supported format, False otherwise.
        """        
        return format is not None and str(format).lower() in self._SUPPORTED_FORMATS
    
    def load_from_file(self, img_path: str) -> None:
        """
        Loads an image from a local file and converts it to ndarray.
        
        Args:
            img_path (str): Path to the image file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedExtensionError: If the file extension is not supported.
        """
        if not isinstance(img_path, str):
            raise ValueError("img_path must be a string")
        
        img_path = Path(img_path).expanduser().resolve()
        if not img_path.is_file():
            raise FileNotFoundError(f"{img_path} doesn't exist")
        
        with Image.open(img_path) as img:
            if not self._is_supported_format(img.format):
                raise UnsupportedExtensionError(f"Unsupported file format: {img.format}")
            
            self._img_array = self._convert_to_grayscale(img)   
        
        self._ascii_array = self._scale = self._advanced_set = None
        
    def load_from_url(self, img_url: str, allow_redirects: bool = False) -> None:
        """
        Loads an image from a URL and converts it to ndarray.
        
        Args:
            img_url (str): URL to the image.
        
        Raises:
            InvalidURLError: If the URL is not valid.
            UnsupportedExtensionError: If the file extension is not supported.
            ImageDownloadError: If the image could not be downloaded.
        """
        if not isinstance(img_url, str):
            raise ValueError("img_url must be a string")
        
        p = urlparse(img_url)
        if p.scheme not in ("http", "https") or not p.netloc:
            raise InvalidURLError(f"{img_url} is not a valid url")
        
        try:
            head = requests.head(img_url, timeout=5, allow_redirects=allow_redirects)
            head.raise_for_status()
        except requests.RequestException as e:
            raise ImageDownloadError(f"Failed to get header: {e}")

        if head.is_redirect and not allow_redirects:
            raise ImageDownloadError("Redirect blocked. Use --allow-redirect")

        content_type = head.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ImageDownloadError("Incorrect Content-Type in response")

        content_length = head.headers.get("Content-Length", 0)
        try:
            content_length = int(content_length)
        except ValueError as e:
            raise ImageDownloadError("Incorrect Content-Length in response")
        
        if content_length and content_length > self._IMAGE_SIZE_LIMIT:
            raise ImageDownloadError("Incorrect file size")
        
        buf = BytesIO()
        try:
            resp = requests.get(img_url, timeout=5, allow_redirects=allow_redirects, stream=True)
            resp.raise_for_status()
            
            for chunk in resp.iter_content(65536):
                buf.write(chunk)
                if buf.tell() > self._IMAGE_SIZE_LIMIT:
                    raise ImageDownloadError("Image exceeds size limit")
                
        except requests.RequestException as e:
            raise ImageDownloadError(f"Download failed: {e}")
        
        buf.seek(0)
        with Image.open(buf) as img:
            try:
                img.verify()
            except (UnidentifiedImageError, DecompressionBombError) as e:
                raise ImageDownloadError(f"Image verification failed: {e}")

        buf.seek(0)
        with Image.open(buf) as img:
            if not self._is_supported_format(img.format):
                raise UnsupportedExtensionError(f"Unsupported file format: {img.format}")
            
            self._img_array = self._convert_to_grayscale(img)    
        
        self._ascii_array = self._scale = self._advanced_set = None
    
    def get_array(self, scale: int = 2, advanced_set: bool = False, skip_rows: bool = False) -> np.ndarray:
        """
        Returns the ASCII image array.
        
        Args:
            scale (int): Determines the size of square blocks (2^scale x 2^scale) used to reduce image resolution.
            advanced_set (bool): Use advanced character set if True.
            skip_rows (bool): If True, uses every second row of pixels, for better display in console.
            
        Returns:
            np.ndarray: ASCII image as a numpy ndarray of characters.
            
        Raises:
            ImageNotLoadedError: If no image is loaded.
        """
        if self._ascii_array is None or scale != self._scale or advanced_set != self._advanced_set:
            self._convert_to_ascii(scale, advanced_set, skip_rows)
    
        return self._ascii_array

    def get_string(self, scale: int = 2, advanced_set: bool = False, skip_rows: bool = True) -> str:
        """
        Returns the ASCII image in form of a string.
        
        Args:
            scale (int): Determines the size of square blocks (2^scale x 2^scale) used to reduce image resolution.
            advanced_set (bool): Use advanced character set if True.
            skip_rows (bool): If True, uses every second row of pixels, for better display in console.
            
        Returns:
            str: ASCII image as a string of characters.
            
        Raises:
            ImageNotLoadedError: If no image is loaded.
        """
        if self._ascii_array is None or scale != self._scale or advanced_set != self._advanced_set:
            self._convert_to_ascii(scale, advanced_set, skip_rows)
    
        return "\n".join("".join(row) for row in self._ascii_array)
    
    def print_ascii(self, scale: int = 2, advanced_set: bool = False, skip_rows: bool = True) -> None:
        """
        Prints the ASCII image to the terminal.
        
        Args:
            scale (int): Determines the size of square blocks (2^scale x 2^scale) used to reduce image resolution.
            advanced_set (bool): Use advanced character set if True.
            skip_rows (bool): If True, uses every second row of pixels, for better display in console.
            
        Raises:
            ImageNotLoadedError: If no image is loaded.
        """
        if self._ascii_array is None or scale != self._scale or advanced_set != self._advanced_set:
            self._convert_to_ascii(scale, advanced_set, skip_rows)
        
        print("\n".join("".join(row) for row in self._ascii_array))
    
    def save_to_img(self, out_path: str = "ascii.png", overwrite: bool = False, scale: int = 2, advanced_set: bool = False, font_size: int = 12, skip_rows: bool = False) -> None:
        """
        Saves the ASCII image as a grayscale image file.
        
        Args:
            out_path (str): Path (including extension) to save the image.
            overwrite (bool): Whether to overwrite preexisting file.
            scale (int): Determines the size of square blocks (2^scale x 2^scale) used to reduce image resolution.
            advanced_set (bool): Use advanced character set if True.
            font_size (int): Font size used to create image.
            skip_rows (bool): If True, uses every second row of pixels, for better display in console.
            
        Raises:
            UnsupportedExtensionError: If the file extension is unsupported.
            TargetExistsError: If the file exists and overwrite is False.
            ImageNotLoadedError: If no image is loaded.
        """
        if not isinstance(overwrite, bool):
            raise ValueError("overwrite must be a boolean")
        
        self._get_font(font_size)
        
        if self._ascii_array is None or scale != self._scale or advanced_set != self._advanced_set:
            self._convert_to_ascii(scale, advanced_set, skip_rows)
        
        out_path = Path(out_path).expanduser().resolve()
        out_format = out_path.suffix.lstrip(".")
        if not self._is_supported_format(out_format):
            raise UnsupportedExtensionError(f"Unsupported file format: {out_format}")
        
        if out_path.is_file() and not overwrite:
            raise TargetExistsError(f"Output file: {out_path} already exist.")

        h = len(self._ascii_array)
        w = max(len(row) for row in self._ascii_array)

        x0, y0, x1, y1 = self._font.getbbox("M")
        cw, ch = x1 - x0, y1 - y0

        img = Image.new("L", (w*cw, h*ch), 255)
        draw = ImageDraw.Draw(img)

        for y, row in enumerate(self._ascii_array):
            for x, ch_ in enumerate(row):
                draw.text((x*cw, y*ch), ch_, font=self._font, fill=0)

        img.save(out_path)
    
    def _get_font(self, font_size: int = 12) -> None:
        """
        Loads a monospaced font for drawing text to an image.
        Tries several common fonts and falls back to the default if none load.
        
        Args:
            font_size (int): Font size used to create image.
        """
        if not isinstance(font_size, int):
            raise ValueError("font_size must be a integer")
        
        fonts = [
            "Courier New.ttf",
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
            "Menlo.ttc",
            "Consolas.ttf"
        ]
        
        for font in fonts:
            try:
                self._font = ImageFont.truetype(font, font_size)
                return
            except (OSError, IOError):
                continue
            
        self._font = ImageFont.load_default()
        
def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="img2ascii",
        description="Converts an image to ASCII art (print or save to file)."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-f", "--file", metavar="PATH",
                     help="path to a local image file")
    src.add_argument("-u", "--url", metavar="URL",
                     help="URL of the image")

    parser.add_argument(
        "-s", "--scale",
        type=int,
        default=2,
        choices=range(0, ImgConverter._MAX_SCALE + 1),
        help="logarithmic scale factor (0–6), default is 2"
    )

    parser.add_argument(
        "-a", "--advanced",
        action="store_true",
        default=False,
        help="use the extended character set, default is False"
    )

    parser.add_argument(
        "-o", "--output", metavar="PATH",
        help="save result to an image file instead of printing to the terminal"
    )

    parser.add_argument(
        "--font-size", type=int, default=12,
        help="font size when saving the file to image, default is 12"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite existing output file"
    )
    
    parser.add_argument(
        "--allow-redirects",
        action="store_true",
        default=False,
        help="allow redirects when using image from url"
    )

    args = parser.parse_args()
    conv = ImgConverter()

    try:
        if args.file:
            conv.load_from_file(args.file)
        else:
            conv.load_from_url(args.url, allow_redirects=args.allow_redirects)

        if args.output:
            conv.save_to_img(
                out_path=args.output,
                overwrite=args.overwrite,
                scale=args.scale,
                advanced_set=args.advanced,
                font_size=args.font_size,
                skip_rows=False
            )
            print(f"Saved to: {args.output}")
        else:
            conv.print_ascii(
                scale=args.scale,
                advanced_set=args.advanced,
                skip_rows=True
            )
    except ImgConverterError as exc:
        parser.error(str(exc))
    