import math
import os
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Final
from urllib.parse import ParseResult, urlparse

import numpy as np
import numpy.typing as npt
import requests
import urllib3
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from PIL.Image import DecompressionBombError


class ImgConverterError(Exception):
    pass


class ImageNotLoadedError(ImgConverterError):
    pass


class InvalidURLError(ImgConverterError):
    pass


class UnsupportedExtensionError(ImgConverterError):
    pass


class ImageDownloadError(ImgConverterError):
    pass


class TargetExistsError(ImgConverterError):
    pass


class ImgConverter:
    _SUPPORTED_FORMATS: Final[tuple[str, ...]] = (
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
    # Pillow plugin IDs; ICO is left out because its plugin decodes pixels already in Image.open
    _INPUT_FORMATS: Final[tuple[str, ...]] = (
        "JPEG",
        "PNG",
        "WEBP",
        "TIFF",
        "BMP",
        "PPM",
    )
    # Both sets go from the densest character (dark pixel) to a space (light pixel)
    _CHARSETS: Final[dict[str, str]] = {
        "simple": "@%#*+=-:. ",
        "advanced": "@$B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvu"
        "nxrjft/\\|()1}{][?-_+~i!lI;:,\"^`'. ",
    }
    MAX_SCALE: Final[int] = 6
    _PIXELS_LIMIT: Final[int] = 8_847_360  # DCI-4K 4096x2160
    _BYTES_PER_PIXEL_WORST: Final[int] = 8  # 4 channels * 2 bytes
    _IMAGE_SIZE_LIMIT: Final[int] = _PIXELS_LIMIT * _BYTES_PER_PIXEL_WORST
    # Pillow's JPEG plugin uses ~60x the file size in memory on a file of empty markers
    _JPEG_SIZE_LIMIT: Final[int] = 30 * 1024 * 1024
    _JPEG_SIGNATURE: Final[bytes] = b"\xff\xd8\xff"
    # Seconds; limit for the whole download and for a single socket read
    _DOWNLOAD_TIMEOUT: Final[int] = 15
    # Lookup table from an 8-bit sRGB value to linear light 0-1, using the sRGB
    # transfer function: a linear segment near black, a 2.4 power curve above it.
    _SRGB_TO_LINEAR: Final[npt.NDArray[np.float32]] = np.where(
        np.arange(256) / 255 <= 0.04045,
        np.arange(256) / 255 / 12.92,
        ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)

    def __init__(self) -> None:
        self._img_array: npt.NDArray[np.uint8] | None = None
        self._font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
        # Cached ASCII result and the arguments it was made with
        self._ascii_array: npt.NDArray[np.str_] | None = None
        self._scale: int | None = None
        self._advanced_set: bool | None = None
        self._skip_rows: bool | None = None
        self._invert: bool | None = None

    def _clear_cache(self) -> None:
        self._ascii_array = None
        self._scale = None
        self._advanced_set = None
        self._skip_rows = None
        self._invert = None

    def _convert_to_grayscale(self, image: Image.Image) -> npt.NDArray[np.uint8]:
        # 1. Decode the image into RGBA, so every input mode has the same 4 channels.
        try:
            rgba_image: Image.Image = image.convert("RGBA")
        except OSError as error:
            raise ImgConverterError(f"Image decoding failed: {error}") from error

        rgba: npt.NDArray[np.float32] = np.asarray(rgba_image, dtype=np.float32)

        # 2. Brightness (luma) as a weighted sum of R, G and B with Rec. 709
        # weights; green counts most because the eye is most sensitive to it.
        luma: npt.NDArray[np.float32] = rgba[..., :3] @ np.array(
            [0.2126, 0.7152, 0.0722], dtype=np.float32
        )

        # 3. Composite on a white background, so semi-transparent edges blend in
        # instead of being drawn as fully opaque: alpha 1 keeps the pixel,
        # alpha 0 gives white (255), values in between mix the two.
        alpha: npt.NDArray[np.float32] = rgba[..., 3] / 255
        luma = luma * alpha + 255 * (1 - alpha)

        # 4. Round (not truncate) to whole 0-255 values.
        return np.round(luma).astype(np.uint8)

    def _convert_to_ascii(
        self, scale: int, advanced_set: bool, skip_rows: bool, invert: bool
    ) -> None:
        # 1. Character set as an array, so a whole array of indices can be
        # turned into characters in one step at the end.
        charset: str = (
            self._CHARSETS["advanced"] if advanced_set else self._CHARSETS["simple"]
        )
        chars: npt.NDArray[np.str_] = np.asarray(list(charset), dtype="<U1")

        # 2. Optional inversion: bright pixels get dense characters, which suits
        # light text on a dark background, e.g. a dark terminal.
        gray: npt.NDArray[np.uint8] = (
            255 - self._img_array if invert else self._img_array
        )

        # 3. sRGB values to linear light through the lookup table. All averaging
        # below is done in linear light; averaging sRGB values directly darkens
        # high-contrast areas.
        linear: npt.NDArray[np.float32] = self._SRGB_TO_LINEAR[gray]

        # 4. Optional row halving: average each pair of rows instead of dropping
        # every second one, so thin horizontal lines do not disappear. An odd
        # last row is cut off first, so the rows pair up exactly.
        if skip_rows:
            even_height: int = linear.shape[0] - linear.shape[0] % 2
            linear = linear[:even_height].reshape(even_height // 2, 2, -1).mean(axis=1)

        # 5. Cut the image down to a whole number of blocks; one block of
        # block_size x block_size pixels becomes one character.
        clamped_scale: int = self._clamp_scale(scale)
        block_size: int = 1 << clamped_scale
        height: int = linear.shape[0] - linear.shape[0] % block_size
        width: int = linear.shape[1] - linear.shape[1] % block_size
        linear = linear[:height, :width]

        # 6. Average each block: the reshape splits rows and columns into
        # (block index, position inside the block), and the mean over the two
        # inside-block axes leaves one value per block.
        if block_size > 1:
            linear = linear.reshape(
                height // block_size, block_size, width // block_size, block_size
            ).mean((1, 3))

        # 7. Back from linear light to 0-255 sRGB values.
        srgb: npt.NDArray[np.uint8] = self._linear_to_srgb(linear)

        # 8. Brightness to character: 256 values split into len(chars) equal-width
        # buckets, darkest bucket gets the first (densest) character.
        char_indices: npt.NDArray[np.uint16] = (
            srgb.astype(np.uint16) * len(chars) // 256
        )
        self._ascii_array = chars[char_indices]

        # 9. Remember the arguments this result was made with, for the cache check.
        self._scale = clamped_scale
        self._advanced_set = advanced_set
        self._skip_rows = skip_rows
        self._invert = invert

    def _linear_to_srgb(self, linear: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        # Inverse of the sRGB transfer function used for _SRGB_TO_LINEAR:
        # a linear segment near black, a 1/2.4 power curve above it.
        srgb: npt.NDArray[np.float32] = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1 / 2.4) - 0.055,
        )
        # Scale 0-1 to 0-255 and round to whole values.
        return np.round(srgb * 255).astype(np.uint8)

    def _clamp_scale(self, scale: int) -> int:
        return max(0, min(scale, self.MAX_SCALE))

    def _validate_conversion_args(
        self, scale: int, advanced_set: bool, skip_rows: bool, invert: bool
    ) -> None:
        # bool is a subclass of int, so True would otherwise pass as scale 1.
        if isinstance(scale, bool) or not isinstance(scale, int):
            raise TypeError("scale must be an integer")

        if not isinstance(advanced_set, bool):
            raise TypeError("advanced_set must be a boolean")

        if not isinstance(skip_rows, bool):
            raise TypeError("skip_rows must be a boolean")

        if not isinstance(invert, bool):
            raise TypeError("invert must be a boolean")

        if self._img_array is None:
            raise ImageNotLoadedError("Load image first")

        clamped_scale: int = self._clamp_scale(scale)
        if clamped_scale != scale:
            # stacklevel=3 points the warning at the caller of the public method, not at this helper.
            warnings.warn(
                f"scale {scale} is out of range 0-{self.MAX_SCALE}, using {clamped_scale}",
                stacklevel=3,
            )

    def _ascii_to_string(self) -> str:
        return "\n".join("".join(row) for row in self._ascii_array)

    def _needs_conversion(
        self, scale: int, advanced_set: bool, skip_rows: bool, invert: bool
    ) -> bool:
        if self._ascii_array is None:
            return True

        # Scales are compared after clamping, so scale 7 reuses a result made for 6.
        return (
            self._clamp_scale(scale) != self._scale
            or advanced_set != self._advanced_set
            or skip_rows != self._skip_rows
            or invert != self._invert
        )

    def _is_supported_format(self, image_format: str | None) -> bool:
        return (
            image_format is not None and image_format.lower() in self._SUPPORTED_FORMATS
        )

    def _check_pixel_limit(self, image: Image.Image) -> None:
        # The size comes from the header, so this runs before any pixels are decoded.
        if image.width * image.height > self._PIXELS_LIMIT:
            raise ImgConverterError(
                f"Image exceeds pixel limit: {image.width}x{image.height}"
            )

    def _check_jpeg_size(self, signature: bytes, size: int) -> None:
        # Must run before Image.open, which allocates memory for every JPEG marker
        # it reads. The signature is checked because Pillow detects the format by
        # content, regardless of the file extension.
        if signature.startswith(self._JPEG_SIGNATURE) and size > self._JPEG_SIZE_LIMIT:
            raise ImgConverterError("JPEG exceeds size limit")

    def load_from_file(self, img_path: str | os.PathLike[str]) -> None:
        if not isinstance(img_path, (str, os.PathLike)):
            raise TypeError("img_path must be a string or a path-like object")

        image_path: Path = Path(img_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{image_path} doesn't exist")

        try:
            file_size: int = image_path.stat().st_size
            if file_size > self._IMAGE_SIZE_LIMIT:
                raise ImgConverterError("Image exceeds size limit")

            with image_path.open("rb") as file:
                self._check_jpeg_size(file.read(len(self._JPEG_SIGNATURE)), file_size)

            with Image.open(image_path, formats=self._INPUT_FORMATS) as image:
                self._check_pixel_limit(image)
                image.verify()
        except (
            UnidentifiedImageError,
            DecompressionBombError,
            SyntaxError,
            OSError,
            ValueError,
        ) as error:
            raise ImgConverterError(f"Image verification failed: {error}") from error

        # Unlike the in-memory buffer in load_from_url, the file can change or disappear between the two opens.
        try:
            with Image.open(image_path, formats=self._INPUT_FORMATS) as image:
                self._check_pixel_limit(image)
                if not self._is_supported_format(image.format):
                    raise UnsupportedExtensionError(
                        f"Unsupported file format: {image.format}"
                    )

                self._img_array = self._convert_to_grayscale(image)
        except (
            UnidentifiedImageError,
            DecompressionBombError,
            SyntaxError,
            OSError,
            ValueError,
        ) as error:
            raise ImgConverterError(f"Image loading failed: {error}") from error

        self._clear_cache()

    def load_from_url(self, img_url: str, allow_redirects: bool = False) -> None:
        if not isinstance(img_url, str):
            raise TypeError("img_url must be a string")

        if not isinstance(allow_redirects, bool):
            raise TypeError("allow_redirects must be a boolean")

        parsed_url: ParseResult = urlparse(img_url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise InvalidURLError(f"{img_url} is not a valid url")

        buffer: BytesIO = BytesIO()
        # Deadline set before the request so connect and header time count too
        deadline: float = time.monotonic() + self._DOWNLOAD_TIMEOUT
        try:
            with requests.get(
                img_url,
                timeout=self._DOWNLOAD_TIMEOUT,
                allow_redirects=allow_redirects,
                stream=True,
            ) as response:
                response.raise_for_status()

                if response.is_redirect and not allow_redirects:
                    raise ImageDownloadError("Redirect blocked. Use --allow-redirects")

                content_type: str = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise ImageDownloadError("Incorrect Content-Type in response")

                content_length_header: str = response.headers.get("Content-Length", "0")
                try:
                    content_length: int = int(content_length_header)
                except ValueError as error:
                    raise ImageDownloadError(
                        "Incorrect Content-Length in response"
                    ) from error

                if content_length and content_length > self._IMAGE_SIZE_LIMIT:
                    raise ImageDownloadError("Incorrect file size")

                # iter_content blocks until a whole chunk arrives, so a server sending
                # one byte at a time would never reach the deadline check. read1
                # returns whatever has arrived; decode_content=True keeps gzip data
                # decoded, so the size limit counts the real image bytes.
                chunk: bytes
                while chunk := response.raw.read1(65536, decode_content=True):
                    buffer.write(chunk)
                    if buffer.tell() > self._IMAGE_SIZE_LIMIT:
                        raise ImageDownloadError("Image exceeds size limit")
                    if time.monotonic() > deadline:
                        raise ImageDownloadError("Download took too long")

        # Errors while reading response.raw come from urllib3, not wrapped by requests.
        except (requests.RequestException, urllib3.exceptions.HTTPError) as error:
            raise ImageDownloadError(f"Download failed: {error}") from error

        self._check_jpeg_size(
            buffer.getbuffer()[: len(self._JPEG_SIGNATURE)].tobytes(), buffer.tell()
        )

        buffer.seek(0)
        try:
            with Image.open(buffer, formats=self._INPUT_FORMATS) as image:
                self._check_pixel_limit(image)
                image.verify()
        except (
            UnidentifiedImageError,
            DecompressionBombError,
            SyntaxError,
            OSError,
            ValueError,
        ) as error:
            raise ImageDownloadError(f"Image verification failed: {error}") from error

        buffer.seek(0)
        with Image.open(buffer, formats=self._INPUT_FORMATS) as image:
            if not self._is_supported_format(image.format):
                raise UnsupportedExtensionError(
                    f"Unsupported file format: {image.format}"
                )

            self._img_array = self._convert_to_grayscale(image)

        self._clear_cache()

    def get_array(
        self,
        scale: int = 2,
        advanced_set: bool = False,
        skip_rows: bool = False,
        invert: bool = False,
    ) -> npt.NDArray[np.str_]:
        self._validate_conversion_args(scale, advanced_set, skip_rows, invert)

        if self._needs_conversion(scale, advanced_set, skip_rows, invert):
            self._convert_to_ascii(scale, advanced_set, skip_rows, invert)

        return self._ascii_array.copy()

    def get_string(
        self,
        scale: int = 2,
        advanced_set: bool = False,
        skip_rows: bool = True,
        invert: bool = False,
    ) -> str:
        self._validate_conversion_args(scale, advanced_set, skip_rows, invert)

        if self._needs_conversion(scale, advanced_set, skip_rows, invert):
            self._convert_to_ascii(scale, advanced_set, skip_rows, invert)

        return self._ascii_to_string()

    def print_ascii(
        self,
        scale: int = 2,
        advanced_set: bool = False,
        skip_rows: bool = True,
        invert: bool = False,
    ) -> None:
        self._validate_conversion_args(scale, advanced_set, skip_rows, invert)

        if self._needs_conversion(scale, advanced_set, skip_rows, invert):
            self._convert_to_ascii(scale, advanced_set, skip_rows, invert)

        print(self._ascii_to_string())

    def save_to_img(
        self,
        out_path: str | os.PathLike[str] = "ascii.png",
        overwrite: bool = False,
        scale: int = 2,
        advanced_set: bool = False,
        font_size: int = 12,
        skip_rows: bool = True,
        invert: bool = False,
    ) -> None:
        if not isinstance(out_path, (str, os.PathLike)):
            raise TypeError("out_path must be a string or a path-like object")

        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")

        # bool is a subclass of int, so True would otherwise pass as font size 1.
        if isinstance(font_size, bool) or not isinstance(font_size, int):
            raise TypeError("font_size must be an integer")

        if font_size <= 0:
            raise ValueError("font_size must be greater than 0")

        self._validate_conversion_args(scale, advanced_set, skip_rows, invert)

        self._get_font(font_size)

        if self._needs_conversion(scale, advanced_set, skip_rows, invert):
            self._convert_to_ascii(scale, advanced_set, skip_rows, invert)

        output_path: Path = Path(out_path).expanduser().resolve()
        out_format: str = output_path.suffix.lstrip(".")
        if not self._is_supported_format(out_format):
            raise UnsupportedExtensionError(f"Unsupported file format: {out_format}")

        if output_path.is_file() and not overwrite:
            raise TargetExistsError(f"Output file: {output_path} already exists.")

        lines: list[str] = ["".join(row) for row in self._ascii_array]

        # Line height is ascent (above the baseline) plus descent (below it), so
        # characters with tails like "p" or "g" do not reach into the next line.
        line_height: int = sum(self._font.getmetrics())
        width: int = math.ceil(max(self._font.getlength(line) for line in lines))

        # White canvas; the text is drawn in black.
        canvas: Image.Image = Image.new("L", (width, len(lines) * line_height), 255)
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)

        # One call per row: the font's own advance spaces the characters.
        row_index: int
        line: str
        for row_index, line in enumerate(lines):
            draw.text((0, row_index * line_height), line, font=self._font, fill=0)

        canvas.save(output_path)

    def _get_font(self, font_size: int = 12) -> None:
        # Common monospaced fonts on Windows, Linux and macOS, tried in order.
        font_names: list[str] = [
            "Courier New.ttf",
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
            "Menlo.ttc",
            "Consolas.ttf",
        ]

        font_name: str
        for font_name in font_names:
            try:
                self._font = ImageFont.truetype(font_name, font_size)
                return
            except OSError:
                continue

        self._font = ImageFont.load_default(size=font_size)
