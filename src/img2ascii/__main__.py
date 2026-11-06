import argparse
import os
import sys

from .img2ascii import ImgConverter, ImgConverterError


def main() -> None:
    """
    Parse the command line arguments, load the image from a file or a URL and
    print the ASCII art or save it to an image file.

    On a conversion, file or network error, a message goes to stderr and the
    process exits with code 1. Invalid arguments are reported by argparse,
    which exits with code 2.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="img2ascii",
        description="Converts an image to ASCII art (print or save to file).",
    )

    source_group: argparse._MutuallyExclusiveGroup = (
        parser.add_mutually_exclusive_group(required=True)
    )
    source_group.add_argument(
        "-f", "--file", metavar="PATH", help="path to a local image file"
    )
    source_group.add_argument("-u", "--url", metavar="URL", help="URL of the image")

    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=2,
        choices=range(ImgConverter.MAX_SCALE + 1),
        help=f"logarithmic scale factor (0-{ImgConverter.MAX_SCALE}), default is 2",
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

    parser.add_argument(
        "--no-invert",
        action="store_true",
        default=False,
        help=(
            "print dark text for a light terminal theme; only affects printing, "
            "a saved image is never inverted"
        ),
    )

    parser.add_argument(
        "-r",
        "--remove-background",
        choices=ImgConverter.BACKGROUND_MODES,
        default=None,
        help=(
            "replace the background with spaces: simple clears the most common "
            "edge characters everywhere, smart only where they touch the top, "
            "left or right edge"
        ),
    )

    args: argparse.Namespace = parser.parse_args()
    # A dark terminal needs inverted brightness, a saved image (black text on
    # white) does not.
    invert: bool = not args.no_invert and not args.output
    converter: ImgConverter = ImgConverter()

    try:
        if args.file:
            converter.load_from_file(args.file)
        else:
            converter.load_from_url(args.url, allow_redirects=args.allow_redirects)

        if args.output:
            converter.save_to_img(
                out_path=args.output,
                overwrite=args.overwrite,
                scale=args.scale,
                advanced_set=args.advanced,
                font_size=args.font_size,
                skip_rows=True,
                invert=invert,
                remove_background=args.remove_background,
            )
            print(f"Saved to: {args.output}")
        else:
            converter.print_ascii(
                scale=args.scale,
                advanced_set=args.advanced,
                skip_rows=True,
                invert=invert,
                remove_background=args.remove_background,
            )

        # Force buffered output out here, so a closed pipe is reported inside
        # this try and not at interpreter exit.
        sys.stdout.flush()
    except BrokenPipeError:
        # Python flushes stdout again on exit, which would raise a second time
        # on the closed pipe.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
    except (ImgConverterError, OSError, ValueError, RuntimeError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
