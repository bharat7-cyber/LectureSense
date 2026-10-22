from pathlib import Path
import re
import sys


TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def normalize_time(value: str) -> str:
    value = value.replace(",", ".")
    if len(value.split(".")[-1]) == 3:
        return value
    return value


def convert_srt_to_mavils(input_path: Path, output_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input SRT not found: {input_path}")

    text = input_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    entries = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines
        if not line:
            i += 1
            continue

        # Standard SRT cue number
        if line.isdigit() and i + 1 < len(lines):
            time_line = lines[i + 1].strip()

            match = TIME_RE.match(time_line)
            if match:
                start = normalize_time(match.group("start"))
                end = normalize_time(match.group("end"))

                i += 2

                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1

                subtitle_text = " ".join(text_lines).strip()

                if subtitle_text:
                    entries.append(
                        f"[{start} --> {end}] {subtitle_text}"
                    )

                continue

        # Also support files without numeric cue indexes
        match = TIME_RE.match(line)
        if match:
            start = normalize_time(match.group("start"))
            end = normalize_time(match.group("end"))

            i += 1

            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            subtitle_text = " ".join(text_lines).strip()

            if subtitle_text:
                entries.append(
                    f"[{start} --> {end}] {subtitle_text}"
                )

            continue

        i += 1

    if not entries:
        raise RuntimeError(
            "No standard SRT entries could be converted. "
            "Check the input file format."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )

    print(f"Converted entries: {len(entries)}")
    print(f"Output: {output_path.resolve()}")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python convert_srt_to_mavils.py "
            "<input.srt> <output.srt>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    convert_srt_to_mavils(input_path, output_path)


if __name__ == "__main__":
    main()