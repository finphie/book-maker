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
from tqdm import tqdm

from library.Ayane.source.shogi.Ayane import UsiEngine, UsiEngineState, UsiThinkResult
from library.sfen import split_ply
from library.yaneuraou import BookPos, EngineOption, UsiThinkResultEncoder

logger = getLogger(__name__)


class MultiThink:
    def __init__(self, output_callback: Optional[Callable[[str, Optional[UsiThinkResult]], None]] = None) -> None:
        self.__sfens: Deque[str] = deque()
        self.__books: Dict[str, List[BookPos]] = {}
        self.__parallel_count: int = 0
        self.__engine_options: EngineOption = EngineOption()
        self.__engine_options.eval_share = True
        self.__go_command_option: str = ''
        self.__engines: List[UsiEngine] = []
        self.__positions: List[str] = []
        self.__output_callback: Callable[[str, Optional[UsiThinkResult]], None] = self.__output if output_callback is None else output_callback

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
        if not engine_path.is_file():
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

        self.__sfens.clear()

        # 解析対象となる局面のみを抽出
        buffer: Dict[str, int] = {}
        for sfen in tqdm(sfens, desc='局面読み込み'):
            trim_sfen, ply = split_ply(sfen)

            # 手数制限
            if not start_moves <= ply <= end_moves:
                continue

            # 局面を追加
            # 重複する局面がある場合、より小さい手数にしておく。
            buffer[trim_sfen] = min(buffer.get(trim_sfen, end_moves), ply)

        for key, value in buffer.items():
            self.__sfens.append(' '.join((key, str(value))))

        logger.info(f'局面数: {len(sfens)}')
        logger.info(f'解析対象局面数: {len(self.__sfens)}')

    def set_books(self, book_path: Optional[Path] = None) -> None:
        self.__books.clear()

        if book_path is None:
            return
        if not book_path.is_file():
            raise FileNotFoundError(f'ファイルが存在しません。: {book_path}')

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
                book_pos = self.__books.get(sfen)

                # 指し手が存在しない場合
                if book_pos is None:
                    self.__books[sfen] = [new_book_pos]
                    continue

                # 重複する指し手は無視
                if any(x.best_move == new_book_pos.best_move for x in book_pos):
                    continue

                self.__books[sfen].append(new_book_pos)

        logger.info(f'定跡局面数: {len(self.__books)}')

    def run(self, *, byoyomi: Optional[int] = None, depth: Optional[int] = None, nodes: Optional[int] = None, cancel_callback: Callable[[], bool] = None) -> None:
        self.__set_go_command_option(byoyomi=byoyomi, depth=depth, nodes=nodes)

        for i, engine in enumerate(self.__engines):
            if not engine.is_connected():
                raise ValueError(f'engine{i}が接続されていません。')

            # 局面の解析を開始
            if not self.__try_analysis(i):
                break

            logger.info(f'engine{i}: 解析開始')
            logger.info(f'- sfen {self.__positions[i]}')

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
                self.__output_callback(self.__positions[i], engine.think_result)

                # 局面の解析を開始
                # 解析対象の局面がない場合は、エンジンを切断する。
                if not self.__try_analysis(i):
                    engine.disconnect()
                    continue

                logger.info(f'engine{i}: 解析開始')
                logger.info(f'- sfen {self.__positions[i]}')

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

    def __output(self, position: str, result: Optional[UsiThinkResult]) -> None:
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
    parser.add_argument('--output', type=Path, default=Path('think.jsonl'), help='出力ファイル名')
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
    output_path: Path = args.output

    if not sfen_path.is_file():
        raise FileNotFoundError(f'ファイルが存在しません。: {sfen_path}')

    sfens = sfen_path.read_text().splitlines()

    def output(position: str, result: Optional[UsiThinkResult]) -> None:
        if result is None:
            return

        data = json.dumps({position: result}, cls=UsiThinkResultEncoder) + '\n'

        with output_path.open('a', encoding='utf_8') as f:
            f.write(data)

    with MultiThink(output_callback=output) as think:
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
