from typing import Tuple


def split_ply(value: str) -> Tuple[str, int]:
    if not value.startswith('sfen '):
        raise ValueError(f'sfen形式の局面ではありません。: {value}')

    x = value[5:].rsplit(' ', 1)
    return x[0], int(x[1])