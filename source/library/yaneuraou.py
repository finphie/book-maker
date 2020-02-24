from dataclasses import asdict, dataclass, field
from distutils.util import strtobool
from json import JSONEncoder
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import psutil

from library.Ayane.source.shogi.Ayane import UsiBound, UsiThinkPV, UsiThinkResult


class RawEngineOption(TypedDict, total=False):
    hash_: str
    threads: str
    book_dir: str
    book_file: str
    eval_dir: str
    network_delay: str
    network_delay2: str
    eval_share: str
    multi_pv: str
    contempt: str
    contempt_from_black: str


@dataclass
class EngineOption:
    __option: RawEngineOption = field(init=False, default_factory=lambda: EngineOption.__get_default())

    @staticmethod
    def __get_default() -> RawEngineOption:
        return RawEngineOption(
            hash_='16',
            threads='1',
            book_dir='book',
            book_file='no_book',
            eval_dir='eval',
            network_delay='0',
            network_delay2='0',
            eval_share='false',
            multi_pv='1',
            contempt='2',
            contempt_from_black='false'
        )

    def to_dict(self) -> Dict[str, str]:
        return asdict(
            self,
            dict_factory=lambda x: {key.replace('_', ''): value.lower() for key, value in x[0][1].items()}
        )

    @property
    def hash(self) -> int:
        return int(self.__option['hash_'])

    @hash.setter
    def hash(self, value: int) -> None:
        if value < 1:
            raise ValueError(f'置換表は1MB以上を指定してください。: {value}')
        if value > psutil.virtual_memory().available:
            raise ValueError(f'置換表には空きメモリ容量未満の数値を指定してください。: {value}')

        self.__option['hash_'] = str(value)

    @property
    def threads(self) -> int:
        return int(self.__option['threads'])

    @threads.setter
    def threads(self, value: int) -> None:
        if value < 1:
            raise ValueError(f'スレッド数は1以上の数値を指定してください。: {value}')

        self.__option['threads'] = str(value)

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
    def eval_dir(self) -> Path:
        return Path(self.__option['eval_dir'])

    @eval_dir.setter
    def eval_dir(self, value: Path) -> None:
        if not value.is_dir():
            raise FileNotFoundError(f'ディレクトリが存在しません。: {value}')

        self.__option['eval_dir'] = str(value)

    @property
    def network_delay(self) -> int:
        return int(self.__option['network_delay'])

    @network_delay.setter
    def network_delay(self, value: int) -> None:
        if value < 0:
            raise ValueError(f'通信の平均遅延時間には0以上の数値を指定してください。: {value}')

        self.__option['network_delay'] = str(value)

    @property
    def network_delay2(self) -> int:
        return int(self.__option['network_delay2'])

    @network_delay2.setter
    def network_delay2(self, value: int) -> None:
        if value < 0:
            raise ValueError(f'通信の最大遅延時間には0以上の数値を指定してください。: {value}')

        self.__option['network_delay2'] = str(value)

    @property
    def eval_share(self) -> bool:
        return strtobool(self.__option['eval_share'])

    @eval_share.setter
    def eval_share(self, value: bool) -> None:
        self.__option['eval_share'] = str(value).lower()

    @property
    def multi_pv(self) -> int:
        return int(self.__option['multi_pv'])

    @multi_pv.setter
    def multi_pv(self, value: int) -> None:
        if value < 1:
            raise ValueError(f'候補手の数には1以上の数値を指定してください。: {value}')

        self.__option['multi_pv'] = str(value)

    @property
    def contempt(self) -> int:
        return int(self.__option['contempt'])

    @contempt.setter
    def contempt(self, value: int) -> None:
        self.__option['contempt'] = str(value)

    @property
    def contempt_from_black(self) -> bool:
        return strtobool(self.__option['contempt_from_black'])

    @contempt_from_black.setter
    def contempt_from_black(self, value: bool) -> None:
        self.__option['contempt_from_black'] = str(value).lower()


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