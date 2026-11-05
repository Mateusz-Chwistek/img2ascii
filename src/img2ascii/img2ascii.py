import requests
import urllib3
import numpy as np
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from PIL.Image import DecompressionBombError
from PIL import Image, ImageFont, ImageDraw, UnidentifiedImageError
import math
import os
import sys
import warnings
import time


class ImgConverterError(Exception):
    """Base ImgConverter exception"""


class ImageNotLoadedError(ImgConverterError):
    """Raised when an image is not loaded properly"""


class InvalidURLError(ImgConverterError):
    """Raised when a provided URL is invalid"""


class UnsupportedExtensionError(ImgConverterError):
    """Raised when the file extension is not supported"""


class ImageDownloadError(ImgConverterError):
    """Raised when an image cannot be downloaded"""


class TargetExistsError(ImgConverterError):
    """Raised when a file already exists at the target location"""


class ImgConverter:
    _SUPPORTED_FORMATS = (
        "jpeg",
        "jpg",
        "png",
        "webp",
        "tif",
        "tiff",
        "bmp",
        "ppm",
        "pgm",
        "pbm",
        "ico",
    )
    _INPUT_FORMATS = (
        "JPEG",
        "PNG",
        "WEBP",
        "TIFF",
        "BMP",
        "PPM",
    )  # Pillow plugin IDs; ICO is left out because its plugin decodes pixels already in Image.open
    _CHARSETS = {
        "simple": "@%#*+=-:. ",
        "advanced": "@$B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvu"
        "nxrjft/\\|()1}{][?-_+~i!lI;:,\"^`'. ",
    }
    _MAX_SCALE = 6
    _PIXELS_LIMIT = 8_847_360  # DCI-4K 4096x2160
    _BYTES_PER_PIXEL_WORST = 8  # 4 channels * 2 bytes
    _IMAGE_SIZE_LIMIT = _PIXELS_LIMIT * _BYTES_PER_PIXEL_WORST
    _JPEG_SIZE_LIMIT = (
        30 * 1024 * 1024
    )  # Pillow's JPEG plugin uses ~60x the file size in memory on a file of empty markers
    _JPEG_SIGNATURE = b"\xff\xd8\xff"
    _DOWNLOAD_TIMEOUT = (
        15  # Seconds; limit for the whole download and for a single socket read
    )

    def __init__(self):
        """Initializes the ImgConverter instance."""
        self._img_array = self._ascii_array = self._font = self._scale = (
            self._advanced_set
        ) = self._skip_rows = None

    def _convert_to_grayscale(self, img: Image.Image):
        """
        Extracts green channel, and zero-out fully transparent pixels.

        Args:
            img (PIL.Image.Image): Input pillow image.

        Returns:
            np.ndarray: 2D array (H x W) of green-channel values, dtype uint8.

        Raises:
            ImgConverterError: If the image data cannot be decoded (wraps OSError).
        """
        try:
            img_rgba = img.convert("RGBA")
        except OSError as e:
            raise ImgConverterError(f"Image decoding failed: {e}") from e

        rgba = np.asarray(img_rgba, dtype=np.uint8)

        luma = np.dot(rgba[..., :3].astype(np.float32), [0.2126, 0.7152, 0.0722])

        alpha = rgba[..., 3]
        luma = np.where(alpha == 0, 255, luma)

        return luma.astype(np.uint8)

    def _convert_to_ascii(
        self, scale: int, advanced_set: bool, skip_rows: bool
    ) -> None:
        """
        Converts the loaded image to an ASCII representation.

        Args:
            scale (int): Power of 2 that defines the size of the block window used for downscaling.
            advanced_set (bool): If True, uses a more detailed character set.
            skip_rows (bool): If True, uses every second row of pixels, for better display in console.
        """
        charset = (
            self._CHARSETS["advanced"] if advanced_set else self._CHARSETS["simple"]
        )
        chars = np.asarray(list(charset), dtype="<U1")

        if skip_rows:
            arr = self._img_array[::2]
        else:
            arr = self._img_array

        h, w = arr.shape
        clamped_scale = self._clamp_scale(scale)
        block = 1 << clamped_scale
        h -= h % block
        w -= w % block
        arr = arr[:h, :w]

        if block > 1:
            arr = (
                arr.reshape(h // block, block, w // block, block)
                .mean((1, 3))
                .astype(np.uint8)
            )

        self._ascii_array = chars[(arr.astype(np.uint16) * (len(chars) - 1) // 255)]
        self._scale = clamped_scale
        self._advanced_set = advanced_set
        self._skip_rows = skip_rows

    def _clamp_scale(self, scale: int) -> int:
        """
        Limits scale to the supported range.

        Args:
            scale (int): Requested scale.

        Returns:
            int: Scale clamped to 0..._MAX_SCALE.
        """
        return max(0, min(scale, self._MAX_SCALE))

    def _validate_conversion_args(
        self, scale: int, advanced_set: bool, skip_rows: bool
    ) -> None:
        """
        Checks the arguments shared by all public conversion methods, that an image is loaded, and warns about an out of range scale.

        Args:
            scale (int): Requested scale.
            advanced_set (bool): Character set flag.
            skip_rows (bool): Row skipping flag.

        Raises:
            TypeError: If `scale` is not an integer or a flag is not a boolean.
            ImageNotLoadedError: If no image is loaded.
        """
        # bool is a subclass of int, so True would otherwise pass as scale 1.
        if isinstance(scale, bool) or not isinstance(scale, int):
            raise TypeError("scale must be an integer")

        if not isinstance(advanced_set, bool):
            raise TypeError("advanced_set must be a boolean")

        if not isinstance(skip_rows, bool):
            raise TypeError("skip_rows must be a boolean")

        if self._img_array is None:
            raise ImageNotLoadedError("Load image first")

        clamped_scale = self._clamp_scale(scale)
        if clamped_scale != scale:
            # stacklevel=3 points the warning at the caller of the public method, not at this helper.
            warnings.warn(
                f"scale {scale} is out of range 0-{self._MAX_SCALE}, using {clamped_scale}",
                stacklevel=3,
            )

    def _needs_conversion(
        self, scale: int, advanced_set: bool, skip_rows: bool
    ) -> bool:
        """
        Checks whether the cached ASCII result was made with different arguments.
        Scales are compared after clamping, so scale 7 reuses a result made for 6.

        Args:
            scale (int): Requested scale.
            advanced_set (bool): Character set flag.
            skip_rows (bool): Row skipping flag.

        Returns:
            bool: True if the ASCII result has to be computed again.
        """
        if self._ascii_array is None:
            return True

        return (
            self._clamp_scale(scale) != self._scale
            or advanced_set != self._advanced_set
            or skip_rows != self._skip_rows
        )

    def _is_supported_format(self, image_format: str | None) -> bool:
        """
        Check if the file has a supported format.

        Args:
            image_format (str | None): File format received from Image.format

        Returns:
            bool: True if the file has a supported format, False otherwise.
        """
        return (
            image_format is not None and image_format.lower() in self._SUPPORTED_FORMATS
        )

    def _check_pixel_limit(self, img: Image.Image) -> None:
        """
        Rejects an image with too many pixels, based on the size read from its header, before any pixels are decoded.

        Args:
            img (PIL.Image.Image): Opened pillow image.

        Raises:
            ImgConverterError: If the image exceeds the pixel limit.
        """
        if img.width * img.height > self._PIXELS_LIMIT:
            raise ImgConverterError(
                f"Image exceeds pixel limit: {img.width}x{img.height}"
            )

    def _check_jpeg_size(self, signature: bytes, size: int) -> None:
        """
        Rejects an oversized JPEG. Must run before Image.open, which allocates memory for every JPEG marker it reads.
        The file signature is used because Pillow detects the format by content, regardless of the file extension.

        Args:
            signature (bytes): First bytes of the image data.
            size (int): Size of the image data in bytes.

        Raises:
            ImgConverterError: If the data is a JPEG that exceeds the JPEG size limit.
        """
        if signature.startswith(self._JPEG_SIGNATURE) and size > self._JPEG_SIZE_LIMIT:
            raise ImgConverterError("JPEG exceeds size limit")

    def load_from_file(self, img_path: str | os.PathLike[str]) -> None:
        """
        Loads an image from a local file and converts it to ndarray.

        Args:
            img_path (str): Path to the image file.

        Raises:
            TypeError: If `img_path` is not a string.
            FileNotFoundError: If the file does not exist.
            UnsupportedExtensionError: If the file extension is not supported.
            ImgConverterError: If the file cannot be read, is not a valid image or exceeds the size or pixel limit.
        """
        if not isinstance(img_path, (str, os.PathLike)):
            raise TypeError("img_path must be a string or a path-like object")

        img_path = Path(img_path).expanduser().resolve()
        if not img_path.is_file():
            raise FileNotFoundError(f"{img_path} doesn't exist")

        try:
            file_size = img_path.stat().st_size
            if file_size > self._IMAGE_SIZE_LIMIT:
                raise ImgConverterError("Image exceeds size limit")

            with img_path.open("rb") as file:
                self._check_jpeg_size(file.read(len(self._JPEG_SIGNATURE)), file_size)

            with Image.open(img_path, formats=self._INPUT_FORMATS) as img:
                self._check_pixel_limit(img)
                img.verify()
        except (
            UnidentifiedImageError,
            DecompressionBombError,
            SyntaxError,
            OSError,
            ValueError,
        ) as e:
            raise ImgConverterError(f"Image verification failed: {e}") from e

        # Unlike the in-memory buffer in load_from_url, the file can change or disappear between the two opens.
        try:
            with Image.open(img_path, formats=self._INPUT_FORMATS) as img:
                self._check_pixel_limit(img)
                if not self._is_supported_format(img.format):
                    raise UnsupportedExtensionError(
                        f"Unsupported file format: {img.format}"
                    )

                self._img_array = self._convert_to_grayscale(img)
        except (
            UnidentifiedImageError,
            DecompressionBombError,
            SyntaxError,
            OSError,
            ValueError,
        ) as e:
            raise ImgConverterError(f"Image loading failed: {e}") from e

        self._ascii_array = self._scale = self._advanced_set = self._skip_rows = None

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
            raise TypeError("img_url must be a string")

        if not isinstance(allow_redirects, bool):
            raise TypeError("allow_redirects must be a boolean")

        p = urlparse(img_url)
        if p.scheme not in ("http", "https") or not p.netloc:
            raise InvalidURLError(f"{img_url} is not a valid url")

        buf = BytesIO()
        # Deadline set before the request so connect and header time count too
        deadline = time.monotonic() + self._DOWNLOAD_TIMEOUT
        try:
            with requests.get(
                img_url,
                timeout=self._DOWNLOAD_TIMEOUT,
                allow_redirects=allow_redirects,
                stream=True,
            ) as resp:
                resp.raise_for_status()

                if resp.is_redirect and not allow_redirects:
                    raise ImageDownloadError("Redirect blocked. Use --allow-redirects")

                content_type = resp.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise ImageDownloadError("Incorrect Content-Type in response")

                content_length = resp.headers.get("Content-Length", 0)
                try:
                    content_length = int(content_length)
                except ValueError as e:
                    raise ImageDownloadError(
                        "Incorrect Content-Length in response"
                    ) from e

                if content_length and content_length > self._IMAGE_SIZE_LIMIT:
                    raise ImageDownloadError("Incorrect file size")

                # iter_content blocks until a whole chunk arrives, so a server sending
                # one byte at a time would never reach the deadline check. read1
                # returns whatever has arrived; decode_content=True keeps gzip data
                # decoded, so the size limit counts the real image bytes.
                while chunk := resp.raw.read1(65536, decode_content=True):
                    buf.write(chunk)
                    if buf.tell() > self._IMAGE_SIZE_LIMIT:
                        raise ImageDownloadError("Image exceeds size limit")
                    if time.monotonic() > deadline:
                        raise ImageDownloadError("Download took too long")

        # Errors while reading resp.raw come from urllib3, not wrapped by requests.
        except (requests.RequestException, urllib3.exceptions.HTTPError) as e:
            raise ImageDownloadError(f"Download failed: {e}") from e

        self._check_jpeg_size(
            buf.getbuffer()[: len(self._JPEG_SIGNATURE)].tobytes(), buf.tell()
        )

        buf.seek(0)
        try:
            with Image.open(buf, formats=self._INPUT_FORMATS) as img:
                self._check_pixel_limit(img)
                img.verify()
        except (
            UnidentifiedImageError,
            DecompressionBombError,
            SyntaxError,
            OSError,
            ValueError,
        ) as e:
            raise ImageDownloadError(f"Image verification failed: {e}") from e

        buf.seek(0)
        with Image.open(buf, formats=self._INPUT_FORMATS) as img:
            if not self._is_supported_format(img.format):
                raise UnsupportedExtensionError(
                    f"Unsupported file format: {img.format}"
                )

            self._img_array = self._convert_to_grayscale(img)

        self._ascii_array = self._scale = self._advanced_set = self._skip_rows = None

    def get_array(
        self, scale: int = 2, advanced_set: bool = False, skip_rows: bool = False
    ) -> np.ndarray:
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
        self._validate_conversion_args(scale, advanced_set, skip_rows)

        if self._needs_conversion(scale, advanced_set, skip_rows):
            self._convert_to_ascii(scale, advanced_set, skip_rows)

        return self._ascii_array.copy()

    def get_string(
        self, scale: int = 2, advanced_set: bool = False, skip_rows: bool = True
    ) -> str:
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
        self._validate_conversion_args(scale, advanced_set, skip_rows)

        if self._needs_conversion(scale, advanced_set, skip_rows):
            self._convert_to_ascii(scale, advanced_set, skip_rows)

        return "\n".join("".join(row) for row in self._ascii_array)

    def print_ascii(
        self, scale: int = 2, advanced_set: bool = False, skip_rows: bool = True
    ) -> None:
        """
        Prints the ASCII image to the terminal.

        Args:
            scale (int): Determines the size of square blocks (2^scale x 2^scale) used to reduce image resolution.
            advanced_set (bool): Use advanced character set if True.
            skip_rows (bool): If True, uses every second row of pixels, for better display in console.

        Raises:
            ImageNotLoadedError: If no image is loaded.
        """
        self._validate_conversion_args(scale, advanced_set, skip_rows)

        if self._needs_conversion(scale, advanced_set, skip_rows):
            self._convert_to_ascii(scale, advanced_set, skip_rows)

        print("\n".join("".join(row) for row in self._ascii_array))

    def save_to_img(
        self,
        out_path: str | os.PathLike[str] = "ascii.png",
        overwrite: bool = False,
        scale: int = 2,
        advanced_set: bool = False,
        font_size: int = 12,
        skip_rows: bool = True,
    ) -> None:
        """
        Saves the ASCII image as a grayscale image file.

        Args:
            out_path (str): Path (including extension) to save the image.
            overwrite (bool): Whether to overwrite preexisting file.
            scale (int): Determines the size of square blocks (2^scale x 2^scale) used to reduce image resolution.
            advanced_set (bool): Use advanced character set if True.
            font_size (int): Font size used to create image.
            skip_rows (bool): If True, uses every second row of pixels, to compensate for character cells being about twice as tall as wide.

        Raises:
            UnsupportedExtensionError: If the file extension is unsupported.
            TargetExistsError: If the file exists and overwrite is False.
            ImageNotLoadedError: If no image is loaded.
        """
        if not isinstance(out_path, (str, os.PathLike)):
            raise TypeError("out_path must be a string or a path-like object")

        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")

        # bool is a subclass of int, so True would otherwise pass as font size 1.
        if isinstance(font_size, bool) or not isinstance(font_size, int):
            raise TypeError("font_size must be an integer")

        if font_size <= 0:
            raise ValueError("font_size must be greater than 0")

        self._validate_conversion_args(scale, advanced_set, skip_rows)

        self._get_font(font_size)

        if self._needs_conversion(scale, advanced_set, skip_rows):
            self._convert_to_ascii(scale, advanced_set, skip_rows)

        out_path = Path(out_path).expanduser().resolve()
        out_format = out_path.suffix.lstrip(".")
        if not self._is_supported_format(out_format):
            raise UnsupportedExtensionError(f"Unsupported file format: {out_format}")

        if out_path.is_file() and not overwrite:
            raise TargetExistsError(f"Output file: {out_path} already exists.")

        h = len(self._ascii_array)
        w = max(len(row) for row in self._ascii_array)

        ascent, descent = self._font.getmetrics()
        cell_width = math.ceil(self._font.getlength("M"))
        cell_height = ascent + descent

        img = Image.new("L", (w * cell_width, h * cell_height), 255)
        draw = ImageDraw.Draw(img)

        for y, row in enumerate(self._ascii_array):
            for x, char in enumerate(row):
                draw.text(
                    (x * cell_width, y * cell_height), char, font=self._font, fill=0
                )

        img.save(out_path)

    def _get_font(self, font_size: int = 12) -> None:
        """
        Loads a monospaced font for drawing text to an image.
        Tries several common fonts and falls back to the default if none load.

        Args:
            font_size (int): Font size used to create image.
        """
        fonts = [
            "Courier New.ttf",
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
            "Menlo.ttc",
            "Consolas.ttf",
        ]

        for font in fonts:
            try:
                self._font = ImageFont.truetype(font, font_size)
                return
            except OSError:
                continue

        self._font = ImageFont.load_default(size=font_size)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="img2ascii",
        description="Converts an image to ASCII art (print or save to file).",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-f", "--file", metavar="PATH", help="path to a local image file")
    src.add_argument("-u", "--url", metavar="URL", help="URL of the image")

    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=2,
        choices=range(0, ImgConverter._MAX_SCALE + 1),
        help="logarithmic scale factor (0–6), default is 2",
    )

    parser.add_argument(
        "-a",
        "--advanced",
        action="store_true",
        default=False,
        help="use the extended character set, default is False",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="save result to an image file instead of printing to the terminal",
    )

    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="font size when saving the file to image, default is 12",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite existing output file",
    )

    parser.add_argument(
        "--allow-redirects",
        action="store_true",
        default=False,
        help="allow redirects when using image from url",
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
                skip_rows=True,
            )
            print(f"Saved to: {args.output}")
        else:
            conv.print_ascii(
                scale=args.scale, advanced_set=args.advanced, skip_rows=True
            )

        # Force buffered output out here, so a closed pipe is reported inside this try and not at interpreter exit.
        sys.stdout.flush()
    except BrokenPipeError:
        # Python flushes stdout again on exit, which would raise a second time on the closed pipe.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
    except (ImgConverterError, OSError, ValueError, RuntimeError) as e:
        print(f"{parser.prog}: error: {e}", file=sys.stderr)
        sys.exit(1)
