import math
import os
import time
import warnings
from collections import deque
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
    """
    Base class for this package's own exceptions.
    """


class ImageNotLoadedError(ImgConverterError):
    """
    Error raised when a conversion is requested before an image is loaded.
    """


class InvalidURLError(ImgConverterError):
    """
    Error raised when a URL is not a plain http or https address.
    """


class UnsupportedExtensionError(ImgConverterError):
    """
    Error raised when an image or an output file uses a format outside the
    supported list.
    """


class ImageDownloadError(ImgConverterError):
    """
    Error raised when an image cannot be downloaded or fails the download
    checks.
    """


class TargetExistsError(ImgConverterError):
    """
    Error raised when the output file already exists and overwrite is off.
    """


class ImgConverter:
    """
    Converter from a loaded image to ASCII art. The result is cached and
    rebuilt only when the arguments change.

    Load an image with ``load_from_file`` or ``load_from_url`` before calling
    the conversion methods.
    """

    #################### Formats ####################

    # Formats accepted after loading (Pillow format IDs, lower case) and output
    # file extensions
    _SUPPORTED_FORMATS: Final[tuple[str, ...]] = (
        "jpeg",
        "jpg",
        "mpo",
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

    # Pillow plugin IDs, ICO is left out because its plugin decodes the pixels
    # already in Image.open
    _INPUT_FORMATS: Final[tuple[str, ...]] = (
        "JPEG",
        "PNG",
        "WEBP",
        "TIFF",
        "BMP",
        "PPM",
    )

    #################### Character sets ####################

    # Both sets go from the densest character (dark pixel) to a space (light pixel)
    _CHARSETS: Final[dict[str, str]] = {
        "simple": "@%#*+=-:. ",
        "advanced": "@$B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvu"
        "nxrjft/\\|()1}{][?-_+~i!lI;:,\"^`'. ",
    }

    #################### Background removal ####################

    # Accepted remove_background modes: "simple" clears the background
    # characters everywhere, "smart" only where they connect to the top, left
    # or right edge
    BACKGROUND_MODES: Final[tuple[str, ...]] = ("simple", "smart")

    # How many of the most common edge characters can count as background
    _BACKGROUND_MAX_CHARS: Final[int] = 3

    # Minimum share of the edge cells a character needs to count as background
    _BACKGROUND_MIN_SHARE: Final[float] = 0.05

    #################### Limits ####################

    # Largest allowed scale, one character averages a 2 ** scale x 2 ** scale
    # pixel block
    MAX_SCALE: Final[int] = 6

    # Maximum total pixel count of the decoded image, DCI-4K 4096x2160
    _PIXELS_LIMIT: Final[int] = 8_847_360

    # Bytes per pixel in the worst case, 4 channels * 2 bytes
    _BYTES_PER_PIXEL_WORST: Final[int] = 8

    # Bytes, the maximum image size accepted from a file or a download
    _IMAGE_SIZE_LIMIT: Final[int] = _PIXELS_LIMIT * _BYTES_PER_PIXEL_WORST

    # Bytes, maximum JPEG size; Pillow's JPEG plugin can use ~60x the file size
    # in memory on a file of empty markers
    _JPEG_SIZE_LIMIT: Final[int] = 30 * 1024 * 1024

    # Magic bytes at the start of every JPEG file, checked before Pillow opens it
    _JPEG_SIGNATURE: Final[bytes] = b"\xff\xd8\xff"

    #################### Download ####################

    # Seconds, limit for the whole download and for a single socket read
    _DOWNLOAD_TIMEOUT: Final[int] = 15

    #################### Color ####################

    # Lookup table from an 8-bit sRGB value to linear light 0-1, using the sRGB
    # transfer function: a linear segment near black, a 2.4 power curve above it.
    _SRGB_TO_LINEAR: Final[npt.NDArray[np.float32]] = np.where(
        np.arange(256) / 255 <= 0.04045,
        np.arange(256) / 255 / 12.92,
        ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)

    def __init__(self) -> None:
        """
        Create an empty converter. Load an image with ``load_from_file`` or
        ``load_from_url`` before converting.
        """

        self._img_array: npt.NDArray[np.uint8] | None = None
        self._font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None

        # Cached ASCII result and the arguments it was made with
        self._ascii_array: npt.NDArray[np.str_] | None = None
        self._scale: int | None = None
        self._advanced_set: bool | None = None
        self._skip_rows: bool | None = None
        self._invert: bool | None = None
        self._remove_background: str | None = None

    def _clear_cache(self) -> None:
        """
        Drop the cached ASCII result and the arguments it was made with.
        Called after every image load, so a stale result is never returned.
        """

        self._ascii_array = None
        self._scale = None
        self._advanced_set = None
        self._skip_rows = None
        self._invert = None
        self._remove_background = None

    def _convert_to_grayscale(self, image: Image.Image) -> npt.NDArray[np.uint8]:
        """
        Decode an image into a 2-D array of 0-255 brightness values.

        :param image: The image to decode, in any mode Pillow supports.
        :type image: Image.Image

        :return: One brightness value per pixel, one row per image row.
        :rtype: npt.NDArray[np.uint8]

        :raises ImgConverterError: When Pillow cannot decode the image.
        """

        # Step 1: Decode the image into RGBA, so every input mode has the same
        # 4 channels.
        try:
            rgba_image: Image.Image = image.convert("RGBA")
        except OSError as error:
            raise ImgConverterError(f"image decoding failed: {error}") from error

        rgba: npt.NDArray[np.float32] = np.asarray(rgba_image, dtype=np.float32)

        # Step 2: Brightness (luma) as a weighted sum of R, G and B with Rec. 709
        # weights; green counts most because the eye is most sensitive to it.
        luma: npt.NDArray[np.float32] = rgba[..., :3] @ np.array(
            [0.2126, 0.7152, 0.0722], dtype=np.float32
        )

        # Step 3: Composite on a white background, so semi-transparent edges blend in
        # instead of being drawn as fully opaque: alpha 1 keeps the pixel,
        # alpha 0 gives white (255), values in between mix the two.
        alpha: npt.NDArray[np.float32] = rgba[..., 3] / 255
        luma = luma * alpha + 255 * (1 - alpha)

        # Step 4: Round (not truncate) to whole 0-255 values.
        return np.round(luma).astype(np.uint8)

    def _convert_to_ascii(
        self,
        scale: int,
        advanced_set: bool,
        skip_rows: bool,
        invert: bool,
        remove_background: str | None,
    ) -> None:
        """
        Rebuild the cached ASCII art from the loaded image, one block of
        2 ** scale x 2 ** scale pixels per character.

        :param scale: Logarithmic scale, clamped to 0-MAX_SCALE here.
        :type scale: int

        :param advanced_set: Use the larger, more detailed character set.
        :type advanced_set: bool

        :param skip_rows: Average each pair of rows before converting, which
            halves the output height.
        :type skip_rows: bool

        :param invert: Flip brightness, for light text on a dark background.
        :type invert: bool

        :param remove_background: Background removal mode from
            BACKGROUND_MODES, or None to keep the background.
        :type remove_background: str | None
        """

        # Step 1: Character set as an array, so a whole array of indices can be
        # turned into characters in one step at the end.
        charset: str = (
            self._CHARSETS["advanced"] if advanced_set else self._CHARSETS["simple"]
        )
        chars: npt.NDArray[np.str_] = np.asarray(list(charset), dtype="<U1")

        # Step 2: Optional inversion: bright pixels get dense characters, which suits
        # light text on a dark background, e.g. a dark terminal.
        gray: npt.NDArray[np.uint8] = (
            255 - self._img_array if invert else self._img_array
        )

        # Step 3: sRGB values to linear light through the lookup table. All averaging
        # below is done in linear light, averaging sRGB values directly darkens
        # high-contrast areas.
        linear: npt.NDArray[np.float32] = self._SRGB_TO_LINEAR[gray]

        # Step 4: Optional row halving: average each pair of rows instead of dropping
        # every second one, so thin horizontal lines do not disappear. An odd
        # last row is cut off first, so the rows pair up exactly.
        if skip_rows:
            even_height: int = linear.shape[0] - linear.shape[0] % 2
            linear = linear[:even_height].reshape(even_height // 2, 2, -1).mean(axis=1)

        # Step 5: Cut the image down to a whole number of blocks, one block of
        # block_size x block_size pixels becomes one character.
        clamped_scale: int = self._clamp_scale(scale)
        block_size: int = 1 << clamped_scale
        height: int = linear.shape[0] - linear.shape[0] % block_size
        width: int = linear.shape[1] - linear.shape[1] % block_size
        linear = linear[:height, :width]

        # Step 6: Average each block: the reshape splits rows and columns into
        # (block index, position inside the block), and the mean over the two
        # inside-block axes leaves one value per block.
        if block_size > 1:
            linear = linear.reshape(
                height // block_size, block_size, width // block_size, block_size
            ).mean((1, 3))

        # Step 7: Back from linear light to 0-255 sRGB values.
        srgb: npt.NDArray[np.uint8] = self._linear_to_srgb(linear)

        # Step 8: Brightness to character: 256 values split into len(chars) equal-width
        # buckets, darkest bucket gets the first (densest) character.
        char_indices: npt.NDArray[np.uint16] = (
            srgb.astype(np.uint16) * len(chars) // 256
        )
        self._ascii_array = chars[char_indices]

        # Step 9: Optional background removal: the characters that dominate the
        # image edges are replaced with spaces.
        if remove_background is not None and self._ascii_array.size > 0:
            self._remove_background_chars(remove_background)

        # Step 10: Remember the arguments this result was made with, for the
        # cache check.
        self._scale = clamped_scale
        self._advanced_set = advanced_set
        self._skip_rows = skip_rows
        self._invert = invert
        self._remove_background = remove_background

    def _find_background_chars(self) -> list[str]:
        """
        Pick the characters that make up the background of the cached ASCII
        art, from its top row and its left and right columns.

        The bottom row is skipped, because a subject often reaches the bottom
        edge. Spaces are skipped, because they are already empty.

        :return: Up to _BACKGROUND_MAX_CHARS characters, most common first.
        :rtype: list[str]
        """

        edges: npt.NDArray[np.str_] = np.concatenate(
            (
                self._ascii_array[0, :],
                self._ascii_array[1:, 0],
                self._ascii_array[1:, -1],
            )
        )
        characters: npt.NDArray[np.str_]
        counts: npt.NDArray[np.int64]
        characters, counts = np.unique(edges, return_counts=True)

        # A character seen on only a few edge cells is more likely part of the
        # subject touching the edge than background.
        min_count: float = self._BACKGROUND_MIN_SHARE * edges.size

        background: list[str] = []
        index: int
        for index in np.argsort(counts, kind="stable")[::-1]:
            if (
                counts[index] < min_count
                or len(background) == self._BACKGROUND_MAX_CHARS
            ):
                break
            if characters[index] != " ":
                background.append(str(characters[index]))

        return background

    def _remove_background_chars(self, mode: str) -> None:
        """
        Replace the background characters of the cached ASCII art with
        spaces.

        :param mode: "simple" clears every occurrence of a background
            character, "smart" only the ones connected to the top, left or
            right edge through other background characters, so areas of the
            same brightness inside the subject stay.
        :type mode: str
        """

        background: list[str] = self._find_background_chars()
        if not background:
            return

        is_background: npt.NDArray[np.bool_] = np.isin(self._ascii_array, background)
        if mode == "simple":
            self._ascii_array[is_background] = " "
            return

        # Flood fill from the edges: a breadth-first search through
        # neighbouring background cells, up, down, left and right. Plain lists
        # are used because indexing them is much faster than indexing numpy
        # arrays one element at a time.
        height: int
        width: int
        height, width = self._ascii_array.shape
        candidates: list[list[bool]] = is_background.tolist()
        reached: list[list[bool]] = [[False] * width for _ in range(height)]
        queue: deque[tuple[int, int]] = deque()

        seeds: list[tuple[int, int]] = (
            [(0, col) for col in range(width)]
            + [(row, 0) for row in range(height)]
            + [(row, width - 1) for row in range(height)]
        )
        row: int
        col: int
        for row, col in seeds:
            if candidates[row][col] and not reached[row][col]:
                reached[row][col] = True
                queue.append((row, col))

        next_row: int
        next_col: int
        while queue:
            row, col = queue.popleft()
            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and candidates[next_row][next_col]
                    and not reached[next_row][next_col]
                ):
                    reached[next_row][next_col] = True
                    queue.append((next_row, next_col))

        self._ascii_array[np.array(reached, dtype=bool)] = " "

    def _linear_to_srgb(self, linear: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """
        Convert linear light values back to 0-255 sRGB values.

        :param linear: Linear light brightness, one value per pixel, 0-1.
        :type linear: npt.NDArray[np.float32]

        :return: The sRGB values, rounded to whole 0-255 numbers.
        :rtype: npt.NDArray[np.uint8]
        """

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
        """
        Clamp a scale into the 0 to ``MAX_SCALE`` range.

        :param scale: The scale to clamp.
        :type scale: int

        :return: The clamped scale.
        :rtype: int
        """

        return max(0, min(scale, self.MAX_SCALE))

    def _validate_conversion_args(
        self,
        scale: int,
        advanced_set: bool,
        skip_rows: bool,
        invert: bool,
        remove_background: str | None,
    ) -> None:
        """
        Check the conversion arguments and reject a conversion that is
        requested before an image is loaded.

        A scale outside 0-MAX_SCALE is clamped and a warning is raised at the
        caller of the public method.

        :param scale: The requested scale.
        :type scale: int

        :param advanced_set: Whether the advanced character set is requested.
        :type advanced_set: bool

        :param skip_rows: Whether row halving is requested.
        :type skip_rows: bool

        :param invert: Whether brightness inversion is requested.
        :type invert: bool

        :param remove_background: The requested background removal mode.
        :type remove_background: str | None

        :raises TypeError: When an argument has the wrong type.
        :raises ValueError: When remove_background is not a known mode.
        :raises ImageNotLoadedError: When no image is loaded yet.
        """

        # bool is a subclass of int, so True would otherwise pass as scale 1.
        if isinstance(scale, bool) or not isinstance(scale, int):
            raise TypeError("scale must be an integer")

        if not isinstance(advanced_set, bool):
            raise TypeError("advanced_set must be a boolean")

        if not isinstance(skip_rows, bool):
            raise TypeError("skip_rows must be a boolean")

        if not isinstance(invert, bool):
            raise TypeError("invert must be a boolean")

        if remove_background is not None:
            if not isinstance(remove_background, str):
                raise TypeError("remove_background must be a string or None")
            if remove_background not in self.BACKGROUND_MODES:
                raise ValueError(
                    f"unknown remove_background mode: {remove_background}, "
                    f"expected one of: {', '.join(self.BACKGROUND_MODES)}"
                )

        if self._img_array is None:
            raise ImageNotLoadedError("no image loaded, load one first")

        clamped_scale: int = self._clamp_scale(scale)
        if clamped_scale != scale:
            # stacklevel=3 points the warning at the caller of the public method,
            # not at this helper.
            warnings.warn(
                f"scale {scale} is out of range 0-{self.MAX_SCALE}, "
                f"using {clamped_scale}",
                stacklevel=3,
            )

    def _ascii_to_string(self) -> str:
        """
        Join the cached ASCII art into one string, one image row per line.

        :return: The ASCII art with a newline between rows.
        :rtype: str

        :raises ValueError: When the image is smaller than one block and the
            result is empty.
        """

        # An image smaller than one block gives no rows or no columns.
        if self._ascii_array.size == 0:
            raise ValueError("image is too small for this scale, the result is empty")

        return "\n".join("".join(row) for row in self._ascii_array)

    def _needs_conversion(
        self,
        scale: int,
        advanced_set: bool,
        skip_rows: bool,
        invert: bool,
        remove_background: str | None,
    ) -> bool:
        """
        Check whether the cached ASCII art still matches the requested
        arguments.

        Scales are compared after clamping, so a scale above MAX_SCALE reuses
        a result made for MAX_SCALE.

        :param scale: The requested scale.
        :type scale: int

        :param advanced_set: Whether the advanced character set is requested.
        :type advanced_set: bool

        :param skip_rows: Whether row halving is requested.
        :type skip_rows: bool

        :param invert: Whether brightness inversion is requested.
        :type invert: bool

        :param remove_background: The requested background removal mode.
        :type remove_background: str | None

        :return: True when ``_convert_to_ascii`` has to run again.
        :rtype: bool
        """

        if self._ascii_array is None:
            return True

        return (
            self._clamp_scale(scale) != self._scale
            or advanced_set != self._advanced_set
            or skip_rows != self._skip_rows
            or invert != self._invert
            or remove_background != self._remove_background
        )

    def _is_supported_format(self, image_format: str | None) -> bool:
        """
        Check a format ID or file extension against the supported list.

        :param image_format: The format Pillow reports for the image, or an
            output file extension.
        :type image_format: str | None

        :return: True when the format is in the supported list.
        :rtype: bool
        """

        return (
            image_format is not None and image_format.lower() in self._SUPPORTED_FORMATS
        )

    def _check_pixel_limit(self, image: Image.Image) -> None:
        """
        Reject an image whose decoded size would exceed the pixel limit.

        :param image: The image to check, its size comes from the header.
        :type image: Image.Image

        :raises ImgConverterError: When the image has more pixels than the
            limit.
        """

        # The size comes from the header, so this runs before any pixels are decoded.
        if image.width * image.height > self._PIXELS_LIMIT:
            raise ImgConverterError(
                f"image exceeds the pixel limit of {self._PIXELS_LIMIT:,} px "
                f"({image.width}x{image.height})"
            )

    def _check_jpeg_size(self, signature: bytes, size: int) -> None:
        """
        Reject a JPEG that is too big to open safely.

        :param signature: The first bytes of the file, used to find JPEGs.
        :type signature: bytes

        :param size: Size of the image data in bytes.
        :type size: int

        :raises ImgConverterError: When the file is a JPEG over the size limit.
        """

        # Must run before Image.open, which allocates memory for every JPEG marker
        # it reads. The signature is checked because Pillow detects the format by
        # content, regardless of the file extension.
        if signature.startswith(self._JPEG_SIGNATURE) and size > self._JPEG_SIZE_LIMIT:
            raise ImgConverterError(
                f"JPEG exceeds the size limit of {self._JPEG_SIZE_LIMIT:,} bytes "
                f"({size:,} bytes)"
            )

    def load_from_file(self, img_path: str | os.PathLike[str]) -> None:
        """
        Load an image from a local file, check its size and verify it before
        decoding the pixels.

        :param img_path: Path to the image file, ~ is expanded.
        :type img_path: str | os.PathLike[str]

        :raises TypeError: When img_path is not a string or a path-like
            object.
        :raises RuntimeError: When the path starts with ~ of an unknown user.
        :raises FileNotFoundError: When the file does not exist.
        :raises ImgConverterError: When the file exceeds the size limit or
            fails verification or decoding.
        :raises UnsupportedExtensionError: When the image format is not
            supported.
        """

        if not isinstance(img_path, (str, os.PathLike)):
            raise TypeError("img_path must be a string or a path-like object")

        image_path: Path = Path(img_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"file not found: {image_path}")

        try:
            file_size: int = image_path.stat().st_size
            if file_size > self._IMAGE_SIZE_LIMIT:
                raise ImgConverterError(
                    f"image exceeds the size limit of {self._IMAGE_SIZE_LIMIT:,} bytes "
                    f"({file_size:,} bytes)"
                )

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
            raise ImgConverterError(f"image verification failed: {error}") from error

        # Unlike the in-memory buffer in load_from_url, the file can change or
        # disappear between the two opens.
        try:
            with Image.open(image_path, formats=self._INPUT_FORMATS) as image:
                self._check_pixel_limit(image)
                if not self._is_supported_format(image.format):
                    raise UnsupportedExtensionError(
                        f"unsupported image format: {image.format}"
                    )

                self._img_array = self._convert_to_grayscale(image)
        except (
            UnidentifiedImageError,
            DecompressionBombError,
            SyntaxError,
            OSError,
            ValueError,
        ) as error:
            raise ImgConverterError(f"image loading failed: {error}") from error

        self._clear_cache()

    def load_from_url(self, img_url: str, allow_redirects: bool = False) -> None:
        """
        Download an image over http or https, check its size and verify it
        before decoding the pixels.

        Proxy and CA bundle environment variables are honoured, credentials
        from ~/.netrc are never sent.

        :param img_url: The URL of the image.
        :type img_url: str

        :param allow_redirects: Follow redirects, they are blocked by default.
        :type allow_redirects: bool

        :raises TypeError: When an argument has the wrong type.
        :raises InvalidURLError: When the URL is not a plain http or https
            address.
        :raises ImageDownloadError: When the download fails or takes longer
            than the download timeout, the response is not an image, the file
            exceeds the size limit or it fails verification.
        :raises ImgConverterError: When the image exceeds the JPEG size or
            pixel limit, or cannot be decoded.
        :raises UnsupportedExtensionError: When the image format is not
            supported.
        """

        if not isinstance(img_url, str):
            raise TypeError("img_url must be a string")

        if not isinstance(allow_redirects, bool):
            raise TypeError("allow_redirects must be a boolean")

        try:
            parsed_url: ParseResult = urlparse(img_url)
        except ValueError as error:
            raise InvalidURLError(f"invalid URL: {img_url}") from error

        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise InvalidURLError(f"invalid URL: {img_url}")

        # With trust_env on, requests silently adds credentials from ~/.netrc
        # for a matching host. Turning it off also drops the proxy and CA bundle
        # environment variables, so those are read here the way requests does
        # and passed on explicitly.
        proxies: dict[str, str] = requests.utils.get_environ_proxies(img_url)
        verify: str | bool = (
            os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
            or True
        )
        session: requests.Session = requests.Session()
        session.trust_env = False

        buffer: BytesIO = BytesIO()
        # Deadline set before the request so connect and header time count too
        deadline: float = time.monotonic() + self._DOWNLOAD_TIMEOUT
        try:
            with (
                session,
                session.get(
                    img_url,
                    timeout=self._DOWNLOAD_TIMEOUT,
                    allow_redirects=allow_redirects,
                    stream=True,
                    proxies=proxies,
                    verify=verify,
                ) as response,
            ):
                response.raise_for_status()

                if response.is_redirect and not allow_redirects:
                    raise ImageDownloadError("redirect blocked, allow_redirects is off")

                content_type: str = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise ImageDownloadError(
                        "response is not an image, Content-Type: "
                        f"{content_type or 'missing'}"
                    )

                content_length_header: str = response.headers.get("Content-Length", "0")
                try:
                    content_length: int = int(content_length_header)
                except ValueError as error:
                    raise ImageDownloadError(
                        f"invalid Content-Length in response: {content_length_header}"
                    ) from error

                if content_length and content_length > self._IMAGE_SIZE_LIMIT:
                    raise ImageDownloadError(
                        f"image exceeds the size limit of "
                        f"{self._IMAGE_SIZE_LIMIT:,} bytes ({content_length:,} bytes)"
                    )

                # iter_content blocks until a whole chunk arrives, so a server sending
                # one byte at a time would never reach the deadline check. read1
                # returns whatever has arrived. decode_content=True keeps gzip data
                # decoded, so the size limit counts the real image bytes.
                chunk: bytes
                while chunk := response.raw.read1(65536, decode_content=True):
                    buffer.write(chunk)
                    if buffer.tell() > self._IMAGE_SIZE_LIMIT:
                        raise ImageDownloadError(
                            f"image exceeds the size limit of "
                            f"{self._IMAGE_SIZE_LIMIT:,} bytes"
                        )
                    if time.monotonic() > deadline:
                        raise ImageDownloadError(
                            f"download took longer than {self._DOWNLOAD_TIMEOUT} s"
                        )

        # Errors while reading response.raw come from urllib3, not wrapped by requests.
        except (requests.RequestException, urllib3.exceptions.HTTPError) as error:
            raise ImageDownloadError(f"download failed: {error}") from error

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
            raise ImageDownloadError(f"image verification failed: {error}") from error

        buffer.seek(0)
        with Image.open(buffer, formats=self._INPUT_FORMATS) as image:
            if not self._is_supported_format(image.format):
                raise UnsupportedExtensionError(
                    f"unsupported image format: {image.format}"
                )

            self._img_array = self._convert_to_grayscale(image)

        self._clear_cache()

    def get_array(
        self,
        scale: int = 2,
        advanced_set: bool = False,
        skip_rows: bool = False,
        invert: bool = False,
        remove_background: str | None = None,
    ) -> npt.NDArray[np.str_]:
        """
        Convert the loaded image to ASCII art and return it as an array.

        The result is cached and reused when the arguments do not change. A
        scale outside 0-MAX_SCALE is clamped and a warning is raised.

        :param scale: Logarithmic scale, the block is 2 ** scale pixels wide.
        :type scale: int

        :param advanced_set: Use the larger, more detailed character set.
        :type advanced_set: bool

        :param skip_rows: Average each pair of rows before converting, which
            halves the array height.
        :type skip_rows: bool

        :param invert: Flip brightness, for light text on a dark background.
        :type invert: bool

        :param remove_background: "simple" or "smart" to replace the
            background with spaces, None to keep it. The background is made
            of the most common characters on the top, left and right edges;
            "simple" clears them everywhere, "smart" only where they
            connect to those edges.
        :type remove_background: str | None

        :return: One character per block, one row per output line, a copy of
            the cached result.
        :rtype: npt.NDArray[np.str_]

        :raises TypeError: When an argument has the wrong type.
        :raises ValueError: When remove_background is not a known mode.
        :raises ImageNotLoadedError: When no image is loaded yet.
        """

        self._validate_conversion_args(
            scale, advanced_set, skip_rows, invert, remove_background
        )

        if self._needs_conversion(
            scale, advanced_set, skip_rows, invert, remove_background
        ):
            self._convert_to_ascii(
                scale, advanced_set, skip_rows, invert, remove_background
            )

        return self._ascii_array.copy()

    def get_string(
        self,
        scale: int = 2,
        advanced_set: bool = False,
        skip_rows: bool = True,
        invert: bool = False,
        remove_background: str | None = None,
    ) -> str:
        """
        Convert the loaded image to ASCII art and return it as one string.

        The result is cached and reused when the arguments do not change. A
        scale outside 0-MAX_SCALE is clamped and a warning is raised.

        :param scale: Logarithmic scale, the block is 2 ** scale pixels wide.
        :type scale: int

        :param advanced_set: Use the larger, more detailed character set.
        :type advanced_set: bool

        :param skip_rows: Average each pair of rows before converting, which
            halves the number of lines.
        :type skip_rows: bool

        :param invert: Flip brightness, for light text on a dark background.
        :type invert: bool

        :param remove_background: "simple" or "smart" to replace the
            background with spaces, None to keep it. The background is made
            of the most common characters on the top, left and right edges;
            "simple" clears them everywhere, "smart" only where they
            connect to those edges.
        :type remove_background: str | None

        :return: The ASCII art with a newline between rows.
        :rtype: str

        :raises TypeError: When an argument has the wrong type.
        :raises ImageNotLoadedError: When no image is loaded yet.
        :raises ValueError: When remove_background is not a known mode, or
            the image is smaller than one block and the result is empty.
        """

        self._validate_conversion_args(
            scale, advanced_set, skip_rows, invert, remove_background
        )

        if self._needs_conversion(
            scale, advanced_set, skip_rows, invert, remove_background
        ):
            self._convert_to_ascii(
                scale, advanced_set, skip_rows, invert, remove_background
            )

        return self._ascii_to_string()

    def print_ascii(
        self,
        scale: int = 2,
        advanced_set: bool = False,
        skip_rows: bool = True,
        invert: bool = False,
        remove_background: str | None = None,
    ) -> None:
        """
        Convert the loaded image to ASCII art and print it to stdout.

        The result is cached and reused when the arguments do not change. A
        scale outside 0-MAX_SCALE is clamped and a warning is raised.

        :param scale: Logarithmic scale, the block is 2 ** scale pixels wide.
        :type scale: int

        :param advanced_set: Use the larger, more detailed character set.
        :type advanced_set: bool

        :param skip_rows: Average each pair of rows before converting, which
            halves the number of lines.
        :type skip_rows: bool

        :param invert: Flip brightness, for light text on a dark background.
        :type invert: bool

        :param remove_background: "simple" or "smart" to replace the
            background with spaces, None to keep it. The background is made
            of the most common characters on the top, left and right edges;
            "simple" clears them everywhere, "smart" only where they
            connect to those edges.
        :type remove_background: str | None

        :raises TypeError: When an argument has the wrong type.
        :raises ImageNotLoadedError: When no image is loaded yet.
        :raises ValueError: When remove_background is not a known mode, or
            the image is smaller than one block and the result is empty.
        :raises OSError: When stdout cannot be written, e.g. a closed pipe.
        """

        self._validate_conversion_args(
            scale, advanced_set, skip_rows, invert, remove_background
        )

        if self._needs_conversion(
            scale, advanced_set, skip_rows, invert, remove_background
        ):
            self._convert_to_ascii(
                scale, advanced_set, skip_rows, invert, remove_background
            )

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
        remove_background: str | None = None,
    ) -> None:
        """
        Convert the loaded image to ASCII art and render it into an image
        file with black text on a white background.

        A scale outside 0-MAX_SCALE is clamped and a warning is raised.

        :param out_path: Path of the file to save, the format comes from the
            extension.
        :type out_path: str | os.PathLike[str]

        :param overwrite: Replace the file when it already exists.
        :type overwrite: bool

        :param scale: Logarithmic scale, the block is 2 ** scale pixels wide.
        :type scale: int

        :param advanced_set: Use the larger, more detailed character set.
        :type advanced_set: bool

        :param font_size: Font size in pixels of the rendered text, ignored
            when no monospaced system font is found.
        :type font_size: int

        :param skip_rows: Average each pair of rows before converting, which
            halves the number of lines.
        :type skip_rows: bool

        :param invert: Flip brightness.
        :type invert: bool

        :param remove_background: "simple" or "smart" to replace the
            background with spaces, None to keep it. The background is made
            of the most common characters on the top, left and right edges;
            "simple" clears them everywhere, "smart" only where they
            connect to those edges.
        :type remove_background: str | None

        :raises TypeError: When an argument has the wrong type.
        :raises ValueError: When font_size is not greater than 0,
            remove_background is not a known mode, or the image is smaller
            than one block and the result is empty.
        :raises RuntimeError: When the path starts with ~ of an unknown user.
        :raises ImageNotLoadedError: When no image is loaded yet.
        :raises UnsupportedExtensionError: When the output format is not
            supported.
        :raises TargetExistsError: When the file exists and overwrite is off.
        :raises OSError: When the file cannot be written, e.g. a missing
            directory or no permission.
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

        self._validate_conversion_args(
            scale, advanced_set, skip_rows, invert, remove_background
        )

        self._get_font(font_size)

        if self._needs_conversion(
            scale, advanced_set, skip_rows, invert, remove_background
        ):
            self._convert_to_ascii(
                scale, advanced_set, skip_rows, invert, remove_background
            )

        output_path: Path = Path(out_path).expanduser().resolve()
        out_format: str = output_path.suffix.lstrip(".")
        if not self._is_supported_format(out_format):
            raise UnsupportedExtensionError(
                f"unsupported output format: {out_format or '(no extension)'}"
            )

        if output_path.is_file() and not overwrite:
            raise TargetExistsError(f"output file already exists: {output_path}")

        # An image smaller than one block gives no rows or no columns, and an
        # empty canvas cannot be saved.
        if self._ascii_array.size == 0:
            raise ValueError("image is too small for this scale, the result is empty")

        lines: list[str] = ["".join(row) for row in self._ascii_array]

        line_height: int
        if isinstance(self._font, ImageFont.FreeTypeFont):
            # Line height is ascent (above the baseline) plus descent (below it),
            # so characters with tails like "p" or "g" do not reach into the
            # next line.
            line_height = sum(self._font.getmetrics())
        else:
            # The bitmap fallback font has no metrics, but every glyph box is
            # the full cell height.
            line_height = self._font.getbbox(" ")[3]
        width: int = math.ceil(max(self._font.getlength(line) for line in lines))

        # White background, the text is drawn in black.
        canvas: Image.Image = Image.new("L", (width, len(lines) * line_height), 255)
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)

        # One call per row: the font's own advance spaces the characters.
        row_index: int
        line: str
        for row_index, line in enumerate(lines):
            draw.text((0, row_index * line_height), line, font=self._font, fill=0)

        canvas.save(output_path)

    def _get_font(self, font_size: int = 12) -> None:
        """
        Load the first monospaced font available on the system, falling back
        to Pillow's built-in bitmap font, which has a single fixed size.

        :param font_size: Font size in pixels.
        :type font_size: int
        """

        # Common monospaced fonts on Windows, Linux and macOS, tried in order.
        # Pillow looks fonts up by file name, and Windows names the files
        # differently from the font (cour.ttf is Courier New).
        font_names: list[str] = [
            "cour.ttf",
            "Courier New.ttf",
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
            "Menlo.ttc",
            "consola.ttf",
            "Consolas.ttf",
        ]

        font_name: str
        for font_name in font_names:
            try:
                self._font = ImageFont.truetype(font_name, font_size)
                return
            except OSError:
                continue

        # load_default returns a proportional font, which would misalign the
        # columns, so the older bitmap font is used even though it ignores
        # font_size.
        self._font = ImageFont.load_default_imagefont()
