from dataclasses import asdict, dataclass, field
from distutils.util import strtobool
from json import JSONEncoder
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import psutil

from library.Ayane.source.shogi.Ayane import UsiBound, UsiThinkPV, UsiThinkResult


class RawEngineOption(TypedDict, total=False):
    threads: str
    hash_: str
    eval_hash: str
    multi_pv: str
    network_delay: str
    network_delay2: str
    contempt: str
    contempt_from_black: str
    eval_dir: str
    book_moves: str
    book_dir: str
    book_file: str
    book_eval_diff: str
    book_eval_black_limit: str
    book_eval_white_limit: str
    book_depth_limit: str


@dataclass
class EngineOption:
    __option: RawEngineOption = field(init=False, default_factory=lambda: EngineOption.__get_default())

    @staticmethod
    def __get_default() -> RawEngineOption:
        return RawEngineOption(
            threads='1',
            hash_='16',
            eval_hash='128',
            multi_pv='1',
            network_delay='0',
            network_delay2='0',
            contempt='2',
            contempt_from_black='false',
            eval_dir='eval',
            book_moves='10000',
            book_dir='book',
            book_file='no_book',
            book_eval_diff='0',
            book_eval_black_limit='-99999',
            book_eval_white_limit='-99999',
            book_depth_limit='0'
        )

    def to_dict(self) -> Dict[str, str]:
        return asdict(
            self,
            dict_factory=lambda x: {key.replace('_', ''): value.lower() for key, value in x[0][1].items()}
        )

    @property
    def threads(self) -> int:
        return int(self.__option['threads'])

    @threads.setter
    def threads(self, value: int) -> None:
        if not 1 <= value <= 512:
            raise ValueError(f'スレッド数には、1以上512以下の数値を指定してください。: {value}')

        self.__option['threads'] = str(value)

    @property
    def hash(self) -> int:
        return int(self.__option['hash_'])

    @hash.setter
    def hash(self, value: int) -> None:
        if not 1 <= value <= min(1048576, psutil.virtual_memory().available / 1024):
            raise ValueError(f'置換表には、1MB以上空きメモリ容量以下の数値を指定してください。: {value}')

        self.__option['hash_'] = str(value)

    @property
    def eval_hash(self) -> int:
        return int(self.__option['eval_hash'])

    @eval_hash.setter
    def eval_hash(self, value: int) -> None:
        if not 1 <= value <= min(1048576, psutil.virtual_memory().available / 1024):
            raise ValueError(f'EvalHashには、1MB以上空きメモリ容量以下の数値を指定してください。: {value}')

        self.__option['eval_hash'] = str(value)

    @property
    def multi_pv(self) -> int:
        return int(self.__option['multi_pv'])

    @multi_pv.setter
    def multi_pv(self, value: int) -> None:
        if not 1 <= value <= 800:
            raise ValueError(f'候補手の数には、1以上800以下の数値を指定してください。: {value}')

        self.__option['multi_pv'] = str(value)

    @property
    def network_delay(self) -> int:
        return int(self.__option['network_delay'])

    @network_delay.setter
    def network_delay(self, value: int) -> None:
        if not 0 <= value <= 10000:
            raise ValueError(f'通信の平均遅延時間には、0以上10000以下の数値を指定してください。: {value}')

        self.__option['network_delay'] = str(value)

    @property
    def network_delay2(self) -> int:
        return int(self.__option['network_delay2'])

    @network_delay2.setter
    def network_delay2(self, value: int) -> None:
        if not 0 <= value <= 10000:
            raise ValueError(f'通信の最大遅延時間には、0以上10000以下の数値を指定してください。: {value}')

        self.__option['network_delay2'] = str(value)

    @property
    def contempt(self) -> int:
        return int(self.__option['contempt'])

    @contempt.setter
    def contempt(self, value: int) -> None:
        if not -30000 <= value <= 30000:
            raise ValueError(f'引き分けを受け入れるスコアには、-30000以上30000以下の数値を指定してください。: {value}')

        self.__option['contempt'] = str(value)

    @property
    def contempt_from_black(self) -> bool:
        return strtobool(self.__option['contempt_from_black'])

    @contempt_from_black.setter
    def contempt_from_black(self, value: bool) -> None:
        self.__option['contempt_from_black'] = str(value).lower()

    @property
    def eval_dir(self) -> Path:
        return Path(self.__option['eval_dir'])

    @eval_dir.setter
    def eval_dir(self, value: Path) -> None:
        if not value.is_dir():
            raise FileNotFoundError(f'ディレクトリが存在しません。: {value}')

        self.__option['eval_dir'] = str(value)

    @property
    def book_moves(self) -> int:
        return int(self.__option['book_moves'])

    @book_moves.setter
    def book_moves(self, value: int) -> None:
        if not 0 <= value <= 10000:
            raise ValueError(f'定跡を用いる手数には、0以上10000以下の数値を指定してください。: {value}')

        self.__option['book_moves'] = str(value)

    @property
    def book_path(self) -> Optional[Path]:
        if self.__option['book_file'] == 'no_book':
            return None

        return Path(self.__option['book_dir']) / Path(self.__option['book_file'])

    @book_path.setter
    def book_path(self, value: Optional[Path]) -> None:
        if value is None:
            self.__option['book_file'] = 'no_book'
            return
        if not value.is_file():
            raise FileNotFoundError(f'ファイルが存在しません。: {value}')

        self.__option['book_dir'] = str(value.parent)
        self.__option['book_file'] = value.name

    @property
    def book_eval_diff(self) -> int:
        return int(self.__option['book_eval_diff'])

    @book_eval_diff.setter
    def book_eval_diff(self, value: int) -> None:
        if not 0 <= value <= 99999:
            raise ValueError(f'定跡の第一候補手との評価値の差には、0以上99999以下の数値を指定してください。: {value}')

        self.__option['book_eval_diff'] = str(value)

    @property
    def book_eval_black_limit(self) -> int:
        return int(self.__option['book_eval_black_limit'])

    @book_eval_black_limit.setter
    def book_eval_black_limit(self, value: int) -> None:
        if not -99999 <= value <= 99999:
            raise ValueError(f'定跡の指し手における先手の評価値下限には、-99999以上99999以下の数値を指定してください。: {value}')

        self.__option['book_eval_black_limit'] = str(value)

    @property
    def book_eval_white_limit(self) -> int:
        return int(self.__option['book_eval_white_limit'])

    @book_eval_white_limit.setter
    def book_eval_white_limit(self, value: int) -> None:
        if not -99999 <= value <= 99999:
            raise ValueError(f'定跡の指し手における後手の評価値下限には、-99999以上99999以下の数値を指定してください。: {value}')

        self.__option['book_eval_white_limit'] = str(value)

    @property
    def book_depth_limit(self) -> int:
        return int(self.__option['book_depth_limit'])

    @book_depth_limit.setter
    def book_depth_limit(self, value: int) -> None:
        if not 0 <= value <= 99999:
            raise ValueError(f'定跡の深さ下限には、0以上99999以下の数値を指定してください。: {value}')

        self.__option['book_depth_limit'] = str(value)


@dataclass(frozen=True)
class BookPos:
    best_move: str
    next_move: str
    value: int
    depth: int
    num: int

    def __str__(self) -> str:
        return ' '.join((self.best_move, self.next_move, str(self.value), str(self.depth), str(self.num)))


class UsiThinkResultEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:  # pylint: disable=method-hidden
        if isinstance(o, UsiThinkResult):
            return o.__dict__
        if isinstance(o, UsiThinkPV):
            return o.__dict__
        if isinstance(o, UsiBound):
            return o.to_string()

        return JSONEncoder.default(self, o)