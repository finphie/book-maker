from __future__ import annotations

import argparse
import json
import time
from collections import deque
from logging import config, getLogger
from pathlib import Path
from types import TracebackType
from typing import Callable, Deque, Dict, List, Optional, Type

import blessed
import psutil
from more_itertools import split_before

from library.Ayane.source.shogi.Ayane import UsiEngine, UsiEngineState, UsiThinkResult
from library.yaneuraou import Book, BookFormatter, EngineOption

logger = getLogger(__name__)


class MultiThink:
    def __init__(self, output_callback: Optional[Callable[[Optional[UsiThinkResult]], None]] = None) -> None:
        self.__sfens: Deque[str] = deque()
        self.__books: Dict[str, Book] = {}
        self.__parallel_count: int = 0
        self.__engine_options: EngineOption = EngineOption()
        self.__engine_options.eval_share = True
        self.__go_command_option: str = ''
        self.__engines: List[UsiEngine] = []
        self.__positions: List[str] = []
        self.__output_callback: Callable[[Optional[UsiThinkResult]], None] = self.__output if output_callback is None else output_callback

    def __enter__(self) -> MultiThink:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]:
        self.disconnect()

    def set_engine_options(self, *, eval_dir: Path = Path('eval'), hash_size: Optional[int] = None, multi_pv: int = 1, contempt: int = 2, contempt_from_black: bool = False) -> None:
        self.__engine_options.eval_dir = eval_dir
        self.__engine_options.hash = int((psutil.virtual_memory().available * 0.75 / 1024 ** 2 - 1024) / self.__parallel_count) if hash_size is None else hash_size
        self.__engine_options.multi_pv = multi_pv
        self.__engine_options.contempt = contempt
        self.__engine_options.contempt_from_black = contempt_from_black

        logger.info('エンジン設定を更新')
        for key, value in self.__engine_options.to_dict().items():
            logger.info(f'- {key}: {value}')

    def init_engine(self, engine_path: Path, parallel_count: Optional[int] = None) -> None:
        if not engine_path.exists():
            raise FileNotFoundError(f'ファイルが存在しません。: {engine_path}')
        if parallel_count is None:
            self.__parallel_count = psutil.cpu_count()
        elif parallel_count >= 1:
            self.__parallel_count = parallel_count
        else:
            raise ValueError(f'並列数には1以上の数値を指定してください。: {parallel_count}')
        if not self.__sfens:
            raise ValueError('思考対象局面が存在しません。')

        logger.info(f'エンジンのパス: {engine_path}')
        engine_options = self.__engine_options.to_dict()

        # 局面数が並列数よりも少ない場合、並列数を局面数に合わせる。
        self.__parallel_count = min(self.__parallel_count, len(self.__sfens))
        logger.info(f'並列数: {self.__parallel_count}')

        self.disconnect()
        self.__engines.clear()
        self.__positions.clear()

        for _ in range(self.__parallel_count):
            engine = UsiEngine()
            engine.set_engine_options(engine_options)
            engine.connect(str(engine_path))
            self.__engines.append(engine)
            self.__positions.append('')

    def set_positions(self, sfens: List[str], start_moves: int = 1, end_moves: int = 1000) -> None:
        if start_moves < 1:
            raise ValueError(f'解析対象とする最小手数には、1以上の数値を指定してください。{start_moves}')
        if start_moves > end_moves:
            raise ValueError(f'解析対象とする最大手数には、最小手数以上の数値を指定してください。{end_moves}')

        # 解析対象となる局面のみを抽出
        self.__sfens.clear()
        for sfen in sfens:
            if not start_moves <= int(sfen.rsplit(' ', 1)[1]) <= end_moves:
                continue
            self.__sfens.append(sfen)

        logger.info(f'局面数: {len(sfens)}')
        logger.info(f'解析対象局面数: {len(self.__sfens)}')

    def set_books(self, book_path: Optional[Path] = None) -> None:
        if book_path is None:
            book_lines = []
        elif book_path.exists():
            book_lines = book_path.read_text(encoding='ascii').splitlines()
            if not book_lines[0] == '#YANEURAOU-DB2016 1.00':
                raise ValueError(f'定跡には、やねうら王定跡フォーマットを利用してください。')
            book_lines = book_lines[1:]
        else:
            raise FileNotFoundError(f'ファイルが存在しません。: {book_path}')

        self.__books.clear()
        for buffer in split_before(book_lines, lambda x: x.startswith('sfen')):
            sfen = buffer[0]
            new_book = BookFormatter.deserialize(buffer)
            new_depth = new_book.body[0].depth
            new_multi_pv = len(new_book.body)
            new_game_ply = new_book.game_ply

            # 同一局面（手数を無視）が存在する場合、以下の順序で格納する定跡を決定する。
            # 1. 探索深さが深い方
            # 2. 探索深さが同じ場合は、候補手数が多い方
            # 3. 探索深さと候補手数の両方が同じ場合は、手数が小さい方
            if sfen in self.__books:
                old_book = self.__books[sfen]
                old_depth = old_book.body[0].depth
                old_multi_pv = len(old_book.body)
                old_game_ply = old_book.game_ply

                if new_depth < old_depth:
                    continue
                if new_depth == old_depth:
                    if new_multi_pv < old_multi_pv:
                        continue
                    if new_multi_pv == old_multi_pv and new_game_ply > old_game_ply:
                        continue

            self.__books[sfen] = new_book

    def run(self, *, byoyomi: Optional[int] = None, depth: Optional[int] = None, nodes: Optional[int] = None, cancel_callback: Callable[[], bool] = None) -> None:
        self.__set_go_command_option(byoyomi=byoyomi, depth=depth, nodes=nodes)

        for i, engine in enumerate(self.__engines):
            if not engine.is_connected():
                raise ValueError(f'engine{i}が接続されていません。')

            # 局面の解析を開始
            if not self.__try_analysis(i):
                break

            logger.info(f'engine{i}: 解析開始')
            logger.info(f'- {self.__positions[i]}')

        while True:
            # 解析を停止するかどうか
            if cancel_callback is not None and cancel_callback():
                self.disconnect()
                logger.info('解析停止')
                return

            for i, engine in enumerate(self.__engines):
                # bestmoveが返ってきているか
                if engine.think_result.bestmove is None:
                    continue

                logger.info(f'engine{i}: 解析完了')

                # 出力
                self.__output_callback(engine.think_result)

                # 局面の解析を開始
                # 解析対象の局面がない場合は、エンジンを切断する。
                if not self.__try_analysis(i):
                    engine.disconnect()
                    continue

                logger.info(f'engine{i}: 解析開始')
                logger.info(f'- {self.__positions[i]}')

            # 全対象局面の解析完了
            if all(x.engine_state == UsiEngineState.Disconnected for x in self.__engines):
                logger.info('解析完了')
                return

            time.sleep(1)

    def disconnect(self) -> None:
        for engine in self.__engines:
            engine.disconnect()

    def clear(self) -> None:
        self.disconnect()
        self.__engines.clear()
        self.__positions.clear()
        self.__sfens.clear()
        self.__books.clear()

    def __set_go_command_option(self, *, byoyomi: Optional[int] = None, depth: Optional[int] = None, nodes: Optional[int] = None) -> None:
        if sum(x is not None for x in (byoyomi, depth, nodes)) != 1:
            raise ValueError(f'秒読み、探索深さ、ノード数のいずれか1つを指定してください。: {byoyomi = } {depth = } {nodes = }')

        logger.info('goコマンド設定を更新')

        if byoyomi is not None and byoyomi > 0:
            self.__go_command_option = f'btime 0 wtime 0 byoyomi {byoyomi}'
            logger.info(f'- 秒読み: {byoyomi}')
            return
        if depth is not None and depth > 0:
            self.__go_command_option = f'depth {depth}'
            logger.info(f'- 探索深さ: {depth}')
            return
        if nodes is not None and nodes > 0:
            self.__go_command_option = f'nodes {nodes}'
            logger.info(f'- ノード数: {nodes}')
            return

        raise ValueError(f'goコマンドの形式が不正です。: {byoyomi = }, {depth = }, {nodes = }')

    def __try_analysis(self, engine_number: int) -> bool:
        if not self.__sfens:
            return False

        if not 0 <= engine_number < len(self.__engines):
            raise ValueError(f'エンジン番号が正しくありません。: {engine_number}')

        position = self.__sfens.popleft()
        self.__positions[engine_number] = position
        engine = self.__engines[engine_number]
        engine.send_command('usinewgame')
        engine.usi_position(position)
        engine.usi_go(self.__go_command_option)

        return True

    def __output(self, result: Optional[UsiThinkResult]) -> None:
        if result is None:
            return

        for i, pv in enumerate(result.pvs):
            logger.info(f'- multipv {i+1} {pv.to_string()}')

        if result.bestmove is not None:
            logger.info(f'- bestmove {result.bestmove}')

        if result.ponder is not None:
            logger.info(f'- ponder {result.ponder}')


if __name__ == '__main__':
    term = blessed.Terminal()

    logger = getLogger('multi_think')
    config.dictConfig(json.loads(Path('logconfig.json').read_text()))

    parser = argparse.ArgumentParser()
    parser.add_argument('engine_path', type=Path, help='やねうら王のパス')
    parser.add_argument('sfen_path', type=Path, help='sfenのパス')
    parser.add_argument('eval_dir', type=Path, help='評価関数のパス')
    parser.add_argument('--book_path', type=Path, help='定跡ファイル')
    parser.add_argument('--start_moves', type=int, default=1, help='解析対象局面とする最小手数')
    parser.add_argument('--end_moves', type=int, default=1000, help='解析対象とする最大手数')
    parser.add_argument('--parallel_count', type=int, help='並列数')
    parser.add_argument('--hash', type=int, help='置換表のサイズ')
    parser.add_argument('--multi_pv', type=int, default=1, help='候補手の数')
    parser.add_argument('--contempt', type=int, default=2, help='引き分けを受け入れるスコア')
    parser.add_argument('--contempt_from_black', action='store_true', help='Contemptを先手番から見た値とします。')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--byoyomi', type=int, help='秒読み')
    group.add_argument('--depth', type=int, help='探索深さ')
    group.add_argument('--nodes', type=int, help='ノード数')

    args = parser.parse_args()
    sfen_path: Path = args.sfen_path

    if not sfen_path.exists():
        raise FileNotFoundError(f'ファイルが存在しません。: {sfen_path}')

    sfens = sfen_path.read_text().splitlines()

    with MultiThink() as think:
        think.set_engine_options(
            eval_dir=args.eval_dir,
            hash_size=args.hash,
            multi_pv=args.multi_pv,
            contempt=args.contempt,
            contempt_from_black=args.contempt_from_black
        )
        think.set_positions(sfens, args.start_moves, args.end_moves)
        think.set_books(args.book_path)
        think.init_engine(args.engine_path, args.parallel_count)

        def cancel() -> bool:
            value = term.inkey(timeout=0.001)
            return value and value.is_sequence and value.name == 'KEY_ESCAPE'

        with term.cbreak():
            think.run(
                byoyomi=args.byoyomi,
                depth=args.depth,
                nodes=args.nodes,
                cancel_callback=cancel
            )
