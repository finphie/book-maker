import argparse
import signal
import threading
import time
from collections import deque
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import psutil
from rx.subject import Subject

from Ayane.source.shogi.Ayane import UsiEngine, UsiThinkPV


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

    def __post_init__(self, book_path: Optional[Path]):
        EngineOption.register_property()

        if book_path is not None:
            self.book_file = book_path.name
            self.book_dir = str(book_path.parent)

    @classmethod
    def register_property(cls):
        if hasattr(cls, 'hash_'):
            return

        cls.hash_ = property(cls.__get_hash, cls.__set_hash)
        cls.threads = property(cls.__get_threads, cls.__set_threads)
        cls.network_delay = property(cls.__get_network_delay, cls.__set_network_delay)
        cls.network_delay2 = property(cls.__get_network_delay2, cls.__set_network_delay2)
        cls.eval_share = property(cls.__get_eval_share, cls.__set_eval_share)
        cls.multi_pv = property(cls.__get_multi_pv, cls.__set_multi_pv)
        cls.contempt = property(cls.__get_contempt, cls.__set_contempt)
        cls.contempt_from_black = property(cls.__get_contempt_from_black, cls.__set_contempt_from_black)

    def to_dict(self):
        return asdict(self, dict_factory=lambda x: {key.replace('_', ''): str(value).lower() for key, value in x})

    def __get_hash(self) -> int:
        return self._hash

    def __set_hash(self, value: int):
        if value < 1:
            raise ValueError(f'置換表は1MB以上を指定してください。: {value}')
        if value > psutil.virtual_memory().available:
            raise ValueError(f'置換表には空きメモリ容量未満の数値を指定してください。: {value}')

        self._hash = value

    def __get_threads(self) -> int:
        return self._threads

    def __set_threads(self, value: int):
        if value < 1:
            raise ValueError(f'スレッド数は1以上の数値を指定してください。: {value}')

        self._threads = value

    def __get_network_delay(self) -> int:
        return self._network_delay

    def __set_network_delay(self, value: int):
        if value < 0:
            raise ValueError(f'通信の平均遅延時間には0以上の数値を指定してください。: {value}')

        self._network_delay = value

    def __get_network_delay2(self) -> int:
        return self._network_delay2

    def __set_network_delay2(self, value: int):
        if value < 0:
            raise ValueError(f'通信の最大遅延時間には0以上の数値を指定してください。: {value}')

        self._network_delay2 = value

    def __get_eval_share(self) -> int:
        return self._eval_share

    def __set_eval_share(self, value: bool):
        self._eval_share = value

    def __get_multi_pv(self) -> int:
        return self._multi_pv

    def __set_multi_pv(self, value: int):
        if value < 1:
            raise ValueError(f'候補手の数には1以上の数値を指定してください。: {value}')

        self._multi_pv = value

    def __get_contempt(self) -> int:
        return self._contempt

    def __set_contempt(self, value: int):
        self._contempt = value

    def __get_contempt_from_black(self) -> bool:
        return self._contempt_from_black

    def __set_contempt_from_black(self, value: bool):
        self._contempt_from_black = value


class MultiThink:
    def __init__(self, sfens: list, book_path: Optional[Path] = None, start_moves: int = 1, end_moves: int = 1000, parallel_count: Optional[int] = None):
        if start_moves < 1:
            raise ValueError(f'思考対象とする最小手数には、1以上の数値を指定してください。{start_moves}')
        if start_moves > end_moves:
            raise ValueError(f'思考対象とする最大手数には、最小手数以上の数値を指定してください。{end_moves}')
        if parallel_count is None:
            self.parallel_count = psutil.cpu_count()
        elif parallel_count >= 1:
            self.parallel_count = parallel_count
        else:
            raise ValueError(f'並列数には1以上の数値を指定してください。: {parallel_count}')

        # 思考対象となるsfenのみを抽出
        self.sfens = deque()
        for sfen in sfens:
            if start_moves > int(sfen.rsplit(' ', 1)[1]) > end_moves:
                continue
            self.sfens.append(sfen)

        self.engine_options = EngineOption(threads=1, book_path=book_path, eval_share=True)
        print(self.engine_options.to_dict())

        self.go_command_option = ''
        self.engines = [UsiEngine() for _ in range(self.parallel_count)]
        self.subject = Subject()
        self.threads: List[threading.Thread] = [None for _ in range(self.parallel_count)]
        self.event = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.stop()

    def set_engine_options(self, eval_dir: str = 'eval', hash_size: Optional[int] = None, multi_pv: int = 1, contempt: int = 2, contempt_from_black: bool = False):
        self.engine_options.eval_dir = eval_dir
        self.engine_options.hash_ = int((psutil.virtual_memory().available * 0.75 / 1024 ** 2 - 1024) / self.parallel_count) if hash_size is None else hash_size
        self.engine_options.multi_pv = multi_pv
        self.engine_options.contempt = contempt
        self.engine_options.contempt_from_black = contempt_from_black

    def init_engine(self, engine_path: Path):
        engine_options = self.engine_options.to_dict()
        print(engine_options)

        for engine in self.engines:
            engine.set_engine_options(engine_options)
            engine.connect(str(engine_path))

    def start(self, byoyomi: int = None, depth: int = None, nodes: int = None):
        self.__set_go_command_option(byoyomi, depth, nodes)
        self.subject.subscribe(self.write_book)

        for i, engine in enumerate(self.engines):
            if not engine.is_connected():
                raise ValueError(f'engine{i}が接続されていません。')

            if not self.__try_analysis(engine):
                self.subject.on_completed()
                break

            if self.threads[i] is None:
                self.threads[i] = threading.Thread(target=self.__worker, args=(i, ))
                self.threads[i].start()

    def stop(self):
        self.event.set()

        for i in range(len(self.threads)):
            if self.threads[i] is None:
                continue
            self.threads[i].join()
            self.threads[i] = None

        for engine in self.engines:
            engine.disconnect()

    def wait(self):
        while not self.event.is_set():
            time.sleep(1)

    def write_book(self, pvs: List[UsiThinkPV]):
        for i, pv in enumerate(pvs):
            print(f'{i} {pv.to_string()}')

    def __set_go_command_option(self, byoyomi: int = None, depth: int = None, nodes: int = None):
        if byoyomi is not None and byoyomi > 0:
            self.go_command_option = f'btime 0 wtime 0 byoyomi {byoyomi}'
            return
        if depth is not None and depth > 0:
            self.go_command_option = f'depth {depth}'
            return
        if nodes is not None and nodes > 0:
            self.go_command_option = f'nodes {nodes}'
            return

        raise ValueError(f'goコマンドの形式が不正です。: {byoyomi = }, {depth = }, {nodes = }')

    def __try_analysis(self, engine: UsiEngine) -> bool:
        if not self.sfens:
            return False

        engine.send_command('usinewgame')
        sfen = self.sfens.popleft()
        engine.usi_position(sfen)
        engine.usi_go(self.go_command_option)

        return True

    def __worker(self, engine_number: int):
        engine = self.engines[engine_number]

        while not self.event.wait(1):
            print(f'worker{engine_number}')

            if engine.think_result.bestmove is None:
                continue

            self.subject.on_next(engine.think_result.pvs)
            if not self.__try_analysis(engine):
                break

        self.subject.on_completed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('engine_path', help='やねうら王のパス')
    parser.add_argument('sfen_path', help='sfenのパス')
    parser.add_argument('eval_dir', help='評価関数のパス')
    parser.add_argument('depth', type=int, help='探索深さ')
    parser.add_argument('--book_path', default='user_book1.db', help='定跡ファイル')
    parser.add_argument('--start_moves', type=int, default=1, help='思考対象局面とする最小手数')
    parser.add_argument('--end_moves', type=int, default=1000, help='思考対象とする最大手数')
    parser.add_argument('--parallel_count', type=int, help='並列数')
    parser.add_argument('--hash', type=int, help='置換表のサイズ')
    parser.add_argument('--multi_pv', type=int, default=1, help='候補手の数')
    parser.add_argument('--contempt', type=int, default=2, help='引き分けを受け入れるスコア')
    parser.add_argument('--contempt_from_black', action='store_true', help='Contemptを先手番から見た値とします。')

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
        think.init_engine(args.engine_path)

        think.start(depth=args.depth)
        think.wait()