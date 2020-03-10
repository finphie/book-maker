from dataclasses import dataclass, field
from enum import Enum, auto
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from library.yaneuraou import BookPos

logger = getLogger(__name__)


def split_sfen(value: str) -> Tuple[str, int]:
    if not value.startswith('sfen '):
        raise ValueError(f'sfen形式の局面ではありません。: {value}')

    position, game_ply = value[5:].rsplit(' ', 1)
    return position, int(game_ply)


def read_books(book_path: Optional[Path] = None) -> None:
    if not book_path.is_file():
        raise FileNotFoundError(f'ファイルが存在しません。: {book_path}')

    books: Dict[str, List[BookPos]] = {}
    version = '#YANEURAOU-DB2016 1.00'
    size = book_path.stat().st_size - len(version) - 2

    with book_path.open(encoding='ascii') as f, tqdm(total=size, desc='定跡ファイル読み込み') as bar:
        if f.readline().rstrip('\n') != version:
            raise ValueError(f'定跡には、やねうら王定跡フォーマットを利用してください。')

        sfen: str = ''

        for line in f:
            # プログレスバーを進める
            bar.update(len(line) + 1)

            # バージョン識別子またはコメントをスキップ
            if line.startswith(('#', '//')):
                continue

            # sfenから始まる局面情報
            if line.startswith('sfen '):
                # 文字列（sfen）と手数を除去
                sfen = line[5:].rsplit(' ', 1)[0]
                continue

            x = line.rstrip('\n').split(' ')
            new_book_pos = BookPos(best_move=x[0], next_move=x[1], value=int(x[2]), depth=int(x[3]), num=int(x[4]))
            book_pos = books.get(sfen)

            # 指し手が存在しない場合
            if book_pos is None:
                books[sfen] = [new_book_pos]
                continue

            # 重複する指し手は無視
            if any(x.best_move == new_book_pos.best_move for x in book_pos):
                continue

            books[sfen].append(new_book_pos)


class GameResult(Enum):
    BLACK_WIN = auto()
    WHITE_WIN = auto()
    SENNICHITE = auto()
    MAX_MOVES = auto()
    UNKNOWN = auto()


@dataclass
class Player:
    name: str = field(init=False)
    rate: int = field(init=False, default=0)


@dataclass
class MoveData:
    move: str
    value: int = field(init=False, default=0)
    pv: str = field(init=False, default='')


@dataclass
class Notation:
    black: Player = field(init=False, default=Player())
    white: Player = field(init=False, default=Player())
    result: GameResult = field(init=False, default=GameResult.UNKNOWN)
    moves: List[MoveData] = field(init=False, default_factory=list)


def read_csa(path: Path) -> Optional[Notation]:  # noqa: C901
    lines = path.read_text().splitlines()

    notation = Notation()

    for line in lines:
        token = line[0]

        # バージョン
        if token == 'V':
            continue

        # 対局者名
        if token == 'N':
            # 先手
            if line[1] == '+':
                notation.black.name = line[2:]
                continue

            # 後手
            if line[1] == '-':
                notation.white.name = line[2:]
                continue

            raise ValueError(f'無効な行: {line}')

        # 棋譜情報
        if token == '$':
            continue

        # 開始局面
        if token == 'P':
            continue

        # 指し手
        if token in {'+', '-'}:
            if len(line) == 1:
                continue

            notation.moves.append(MoveData(move=line))
            continue

        # 消費時間
        if token == 'T':
            continue

        # 終局状況
        if token == '%':
            result = line[1:]

            # 投了または入玉宣言
            if result in {'TORYO', 'KACHI'}:
                notation.result = GameResult.WHITE_WIN if len(notation.moves) % 2 == 0 else GameResult.BLACK_WIN
                continue

            # 千日手
            if result == 'SENNICHITE':
                notation.result = GameResult.SENNICHITE
                continue

            # 最大手数制限
            if result == 'MAX_MOVES':
                notation.result = GameResult.MAX_MOVES
                continue

            # それ以外
            notation.result = GameResult.UNKNOWN
            continue

        # コメント
        if token == "'":
            comment = line[1:]

            # 評価値と読み筋
            if comment.startswith('** '):
                x = comment[3:].split(' ', 1)
                notation.moves[-1].value = int(x[0])

                # 読み筋がある場合
                if len(x) == 2:
                    notation.moves[-1].pv = x[1]

                continue

            # 先手のレーティング
            if comment.startswith('black_rate:'):
                notation.black.rate = int(float(line.split(':')[-1]))
                continue

            # 後手のレーティング
            if comment.startswith('white_rate:'):
                notation.white.rate = int(float(line.split(':')[-1]))
                continue

            # floodgate summary行
            if comment.startswith('summary:'):
                result, _, _ = comment[8:].split(':')
                if result == 'max_moves':
                    notation.result = GameResult.MAX_MOVES
                continue

            continue

        raise ValueError(f'無効な行: {line}')

    return notation
