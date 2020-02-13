from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections import deque
from dataclasses import InitVar, asdict, dataclass, field
from logging import config, getLogger
from pathlib import Path
from types import TracebackType
from typing import Callable, Dict, List, Optional, Type

import psutil
from rx.subject import Subject

from Ayane.source.shogi.Ayane import UsiEngine, UsiThinkResult

logger = getLogger(__name__)


@dataclass
class EngineOption:
    hash_: int = 16
    threads: int = 1
    book_path: InitVar[Optional[Path]] = None
    book_file: str = field(default='no_book', init=False)
    book_dir: str = field(default='book', init=False)
    eval_dir: str = 'eval'
    network_delay: int = 0
    network_delay2: int = 0
    eval_share: bool = False
    multi_pv: int = 1
    contempt: int = 2
    contempt_from_black: bool = False

    def __post_init__(self, book_path: Optional[Path]) -> None:
        EngineOption.register_property()

        if book_path is not None:
            self.book_file = book_path.name
            self.book_dir = str(book_path.parent)

    @classmethod
    def register_property(cls) -> None:
        if hasattr(cls, 'hash_'):
            return

        cls.hash_ = property(cls.__get_hash, cls.__set_hash) # type: ignore # noqa: E261
        cls.threads = property(cls.__get_threads, cls.__set_threads) # type: ignore # noqa: E261
        cls.network_delay = property(cls.__get_network_delay, cls.__set_network_delay)# type: ignore # noqa: E261
        cls.network_delay2 = property(cls.__get_network_delay2, cls.__set_network_delay2)# type: ignore # noqa: E261
        cls.eval_share = property(cls.__get_eval_share, cls.__set_eval_share) # type: ignore # noqa: E261
        cls.multi_pv = property(cls.__get_multi_pv, cls.__set_multi_pv) # type: ignore # noqa: E261
        cls.contempt = property(cls.__get_contempt, cls.__set_contempt) # type: ignore # noqa: E261
        cls.contempt_from_black = property(cls.__get_contempt_from_black, cls.__set_contempt_from_black) # type: ignore # noqa: E261

    def to_dict(self) -> Dict[str, str]:
        return asdict(self, dict_factory=lambda x: {key.replace('_', ''): str(value).lower() for key, value in x})

    def __get_hash(self) -> int:
        return self._hash

    def __set_hash(self, value: int) -> None:
        if value < 1:
            raise ValueError(f'置換表は1MB以上を指定してください。: {value}')
        if value > psutil.virtual_memory().available:
            raise ValueError(f'置換表には空きメモリ容量未満の数値を指定してください。: {value}')

        self._hash = value

    def __get_threads(self) -> int:
        return self._threads

    def __set_threads(self, value: int) -> None:
        if value < 1:
            raise ValueError(f'スレッド数は1以上の数値を指定してください。: {value}')

        self._threads = value

    def __get_network_delay(self) -> int:
        return self._network_delay

    def __set_network_delay(self, value: int) -> None:
        if value < 0:
            raise ValueError(f'通信の平均遅延時間には0以上の数値を指定してください。: {value}')

        self._network_delay = value

    def __get_network_delay2(self) -> int:
        return self._network_delay2

    def __set_network_delay2(self, value: int) -> None:
        if value < 0:
            raise ValueError(f'通信の最大遅延時間には0以上の数値を指定してください。: {value}')

        self._network_delay2 = value

    def __get_eval_share(self) -> int:
        return self._eval_share

    def __set_eval_share(self, value: bool) -> None:
        self._eval_share = value

    def __get_multi_pv(self) -> int:
        return self._multi_pv

    def __set_multi_pv(self, value: int) -> None:
        if value < 1:
            raise ValueError(f'候補手の数には1以上の数値を指定してください。: {value}')

        self._multi_pv = value

    def __get_contempt(self) -> int:
        return self._contempt

    def __set_contempt(self, value: int) -> None:
        self._contempt = value

    def __get_contempt_from_black(self) -> bool:
        return self._contempt_from_black

    def __set_contempt_from_black(self, value: bool) -> None:
        self._contempt_from_black = value


class MultiThink:
    def __init__(self, sfens: list, book_path: Optional[Path] = None, start_moves: int = 1, end_moves: int = 1000, parallel_count: Optional[int] = None, output_callback: Optional[Callable[[UsiThinkResult], None]] = None) -> None:
        if start_moves < 1:
            raise ValueError(f'解析対象とする最小手数には、1以上の数値を指定してください。{start_moves}')
        if start_moves > end_moves:
            raise ValueError(f'解析対象とする最大手数には、最小手数以上の数値を指定してください。{end_moves}')
        if parallel_count is None:
            self.__parallel_count = psutil.cpu_count()
        elif parallel_count >= 1:
            self.__parallel_count = parallel_count
        else:
            raise ValueError(f'並列数には1以上の数値を指定してください。: {parallel_count}')

        # 解析対象となるsfenのみを抽出
        self.__sfens = deque()
        for sfen in sfens:
            if start_moves > int(sfen.rsplit(' ', 1)[1]) > end_moves:
                continue
            self.__sfens.append(sfen)

        logger.info(f'局面数: {len(sfens)}')
        logger.info(f'解析対象局面数: {len(self.__sfens)}')
        logger.info(f'並列数: {self.__parallel_count}')

        self.__engine_options = EngineOption(threads=1, book_path=book_path, eval_share=True)
        self.__go_command_option = ''
        self.__engines = [UsiEngine() for _ in range(self.__parallel_count)]
        self.__subject = Subject()
        self.__threads: List[Optional[threading.Thread]] = [None for _ in range(self.__parallel_count)]
        self.__event = threading.Event()
        self.__output_callback = self.__output if output_callback is None else output_callback

    def __enter__(self) -> MultiThink:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]:
        self.stop()

    def set_engine_options(self, eval_dir: str = 'eval', hash_size: Optional[int] = None, multi_pv: int = 1, contempt: int = 2, contempt_from_black: bool = False) -> None:
        self.__engine_options.eval_dir = eval_dir
        self.__engine_options.hash_ = int((psutil.virtual_memory().available * 0.75 / 1024 ** 2 - 1024) / self.__parallel_count) if hash_size is None else hash_size
        self.__engine_options.multi_pv = multi_pv
        self.__engine_options.contempt = contempt
        self.__engine_options.contempt_from_black = contempt_from_black

        logger.info('エンジン設定を更新')
        for key, value in self.__engine_options.to_dict().items():
            logger.info(f'- {key}: {value}')

    def init_engine(self, engine_path: Path) -> None:
        if not engine_path.exists():
            raise FileNotFoundError(f'ファイルが存在しません。: {engine_path}')

        logger.info(f'エンジンのパス: {engine_path}')
        engine_options = self.__engine_options.to_dict()

        for engine in self.__engines:
            engine.set_engine_options(engine_options)
            engine.connect(str(engine_path))

    def start(self, byoyomi: Optional[int] = None, depth: Optional[int] = None, nodes: Optional[int] = None) -> None:
        if sum(x is not None for x in (byoyomi, depth, nodes)) != 1:
            raise ValueError(f'秒読み、探索深さ、ノード数のいずれか1つを指定してください。: {byoyomi = } {depth = } {nodes = }')

        self.__set_go_command_option(byoyomi, depth, nodes)
        self.__subject.subscribe(self.__output_callback)

        for i, engine in enumerate(self.__engines):
            if not engine.is_connected():
                raise ValueError(f'engine{i}が接続されていません。')

            if not self.__try_analysis(engine):
                self.__subject.on_completed()
                break
            logger.info(f'worker{i}: 解析開始')

            if self.__threads[i] is None:
                self.__threads[i] = threading.Thread(target=self.__worker, args=(i, ))
                self.__threads[i].start()

    def stop(self) -> None:
        self.__event.set()

        for i in range(len(self.__threads)):
            if self.__threads[i] is None:
                continue
            self.__threads[i].join()
            self.__threads[i] = None

        for engine in self.__engines:
            engine.disconnect()

    def wait(self) -> None:
        while not self.__event.is_set():
            time.sleep(1)

    def __set_go_command_option(self, byoyomi: int = None, depth: int = None, nodes: int = None) -> None:
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

    def __try_analysis(self, engine: UsiEngine) -> bool:
        if not self.__sfens:
            logger.info('全対象局面の解析完了')
            return False

        engine.send_command('usinewgame')
        sfen = self.__sfens.popleft()
        engine.usi_position(sfen)
        engine.usi_go(self.__go_command_option)

        return True

    def __worker(self, engine_number: int) -> None:
        worker_name = f'worker{engine_number}: '
        logger.info(worker_name + '開始')
        engine = self.__engines[engine_number]

        while not self.__event.wait(1):
            if engine.think_result.bestmove is None:
                continue

            logger.info(worker_name + '解析完了')
            self.__subject.on_next(engine.think_result)
            if not self.__try_analysis(engine):
                break
            logger.info(worker_name + '解析開始')

        self.__subject.on_completed()
        logger.info(worker_name + '終了')

    def __output(self, result: UsiThinkResult) -> None:
        logger.info(result.to_string().replace('\n', ','))


if __name__ == '__main__':
    logger = getLogger('multi_think')
    config.dictConfig(json.loads(Path('logconfig.json').read_text()))

    parser = argparse.ArgumentParser()
    parser.add_argument('engine_path', help='やねうら王のパス')
    parser.add_argument('sfen_path', help='sfenのパス')
    parser.add_argument('eval_dir', help='評価関数のパス')
    parser.add_argument('--book_path', default='user_book1.db', help='定跡ファイル')
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
    sfens = Path(args.sfen_path).read_text().splitlines()

    with MultiThink(sfens, Path(args.book_path), args.start_moves, args.end_moves, args.parallel_count) as think:
        signal.signal(signal.SIGINT, lambda number, frame: think.stop())

        think.set_engine_options(
            eval_dir=args.eval_dir,
            hash_size=args.hash,
            multi_pv=args.multi_pv,
            contempt=args.contempt,
            contempt_from_black=args.contempt_from_black
        )
        think.init_engine(Path(args.engine_path))

        think.start(byoyomi=args.byoyomi, depth=args.depth, nodes=args.nodes)
        think.wait()
