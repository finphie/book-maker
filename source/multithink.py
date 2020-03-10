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
from library.core import split_sfen
from library.yaneuraou import EngineOption, UsiThinkResultEncoder

logger = getLogger(__name__)


class MultiThink:
    def __init__(self, output_callback: Optional[Callable[[str, Optional[UsiThinkResult]], None]] = None) -> None:
        self.__sfens: Deque[str] = deque()
        self.__parallel_count: int = 0
        self.__engine_options: EngineOption = EngineOption()
        self.__go_command_option: str = ''
        self.__engines: List[UsiEngine] = []
        self.__positions: List[str] = []
        self.__output_callback: Callable[[str, Optional[UsiThinkResult]], None] = self.__output if output_callback is None else output_callback

    def __enter__(self) -> MultiThink:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]:
        self.disconnect()

    def set_engine_options(self, *, hash_size: Optional[int] = None, multi_pv: int = 1, contempt: int = 2, contempt_from_black: bool = False, eval_dir: Path = Path('eval'), book_path: Optional[Path] = None) -> None:
        self.__engine_options.hash = int((psutil.virtual_memory().available * 0.75 / 1024 ** 2 - 1024) / self.__parallel_count) if hash_size is None else hash_size
        self.__engine_options.multi_pv = multi_pv
        self.__engine_options.contempt = contempt
        self.__engine_options.contempt_from_black = contempt_from_black
        self.__engine_options.eval_dir = eval_dir
        self.__engine_options.book_path = book_path

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
            trim_sfen, ply = split_sfen(sfen)

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

    def __set_go_command_option(self, *, byoyomi: Optional[int] = None, depth: Optional[int] = None, nodes: Optional[int] = None) -> None:
        logger.info('goコマンド設定を更新')
        go_command_options: List[str] = []

        if byoyomi is not None and byoyomi > 0:
            go_command_options.append(f'btime 0 wtime 0 byoyomi {byoyomi}')
            logger.info(f'- 秒読み: {byoyomi}')
        if depth is not None and depth > 0:
            go_command_options.append(f'depth {depth}')
            logger.info(f'- 探索深さ: {depth}')
        if nodes is not None and nodes > 0:
            go_command_options.append(f'nodes {nodes}')
            logger.info(f'- ノード数: {nodes}')

        if len(go_command_options) == 0:
            raise ValueError(f'goコマンドの形式が不正です。: {byoyomi = }, {depth = }, {nodes = }')

        self.__go_command_option = ' '.join(go_command_options)

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

    go_command_option_group = parser.add_argument_group('goコマンド設定')
    go_command_option_group.add_argument('--byoyomi', type=int, help='秒読み')
    go_command_option_group.add_argument('--depth', type=int, help='探索深さ')
    go_command_option_group.add_argument('--nodes', type=int, help='ノード数')

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

    with MultiThink(output) as think, term.cbreak():
        think.set_engine_options(
            hash_size=args.hash,
            multi_pv=args.multi_pv,
            contempt=args.contempt,
            contempt_from_black=args.contempt_from_black,
            eval_dir=args.eval_dir,
            book_path=args.book_path,
        )
        think.set_positions(sfens, args.start_moves, args.end_moves)
        think.init_engine(args.engine_path, args.parallel_count)

        def cancel() -> bool:
            value = term.inkey(timeout=0.001)
            return value and value.is_sequence and value.name == 'KEY_ESCAPE'

        think.run(
            byoyomi=args.byoyomi,
            depth=args.depth,
            nodes=args.nodes,
            cancel_callback=cancel
        )