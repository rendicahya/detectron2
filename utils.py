from contextlib import nullcontext
from pathlib import Path

from tqdm import tqdm


def iterate(path: Path, operation, extension=None, progress_bar=True, single=False):
    n_files = count_files(path, recursive=True, extension=extension)

    with tqdm(total=n_files) if progress_bar else nullcontext() as bar:
        for action in path.iterdir():
            for video in action.iterdir():
                if video.suffix != extension:
                    continue

                if progress_bar:
                    bar.set_description(video.name[:30])
                    bar.update(1)

                operation(action, video)

                if single:
                    break

            if single:
                break


def count_files(path: Path, recursive=True, extension=None):
    pattern = "**/*" if recursive else "*"

    if extension is not None:
        pattern += "." + extension

    return sum(1 for f in path.glob(pattern) if f.is_file())
