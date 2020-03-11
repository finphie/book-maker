import argparse
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np
import shogi
import shogi.CSA
from shogi.CSA import Parser
from tqdm import tqdm

from library.core import GameResult, read_csa, split_sfen


class MakeSfen:
    __MAX_VALUE = 100000

    def __init__(self, csa_path: Path) -> None:
        self.__csa_files: Generator[Path, None, None]
        if csa_path.is_dir():
            self.__csa_files = csa_path.glob('**/*.csa')
        elif csa_path.is_file():
            def x() -> Generator[Path, None, None]:
                yield csa_path
            self.__csa_files = x()
        else:
            raise ValueError(f'csaファイルが存在しません。: {csa_path}')

        # 最小レーティング
        self.__min_rate: int = 0

        # 最大レーティング
        self.__max_rate: int = 10000

        # 先手の最小評価値
        self.__min_black_value: int = -MakeSfen.__MAX_VALUE

        # 先手の最大評価値
        self.__max_black_value: int = MakeSfen.__MAX_VALUE

        # 後手の最小評価値
        self.__min_white_value: int = -MakeSfen.__MAX_VALUE

        # 後手の最大評価値
        self.__max_white_value: int = MakeSfen.__MAX_VALUE

        # 千日手での最小評価値
        # Noneの時は、先後の最小評価値に従う。
        self.__min_draw_value: Optional[int] = None

        # 千日手での最大評価値
        self.__max_draw_value: int = MakeSfen.__MAX_VALUE

        # 最大差分評価値
        self.__max_diff_value: int = MakeSfen.__MAX_VALUE

        # 出力対象とする最大評価値
        self.__end_value: int = MakeSfen.__MAX_VALUE

    def set_rate_limit(self, min_rate: int, max_rate: int) -> None:
        if min_rate < 0:
            raise ValueError(f'最低レーティングには、0以上の数値を指定してください。: {min_rate}')
        if max_rate < min_rate:
            raise ValueError(f'最高レーティングには、最低レーティング以上の数値を指定してください。: {max_rate}')

        self.__min_rate = min_rate
        self.__max_rate = max_rate

    def set_result_filter(self, result: GameResult, min_value: Optional[int], max_value: Optional[int]) -> None:
        if min_value is max_value is not None and min_value > max_value:
            raise ValueError(f'最小評価値は最大評価値以下の数値を指定してください。: {min_value}')

        if result == GameResult.BLACK_WIN:
            self.__min_black_value = -MakeSfen.__MAX_VALUE if min_value is None else min_value
            self.__max_black_value = MakeSfen.__MAX_VALUE if max_value is None else max_value
        elif result == GameResult.WHITE_WIN:
            self.__min_white_value = -MakeSfen.__MAX_VALUE if max_value is None else -max_value
            self.__max_white_value = MakeSfen.__MAX_VALUE if min_value is None else -min_value
        elif result == GameResult.SENNICHITE:
            self.__min_draw_value = min_value
            self.__max_draw_value = MakeSfen.__MAX_VALUE if max_value is None else max_value

    def set_diff_value_limit(self, diff: int) -> None:
        self.__max_diff_value = diff

    def set_value_limit(self, end_value: int) -> None:
        self.__end_value = end_value

    def run(self) -> List[str]:  # noqa: C901
        sfens: Dict[str, int] = {'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -': 1}

        for csa_file in tqdm(self.__csa_files):
            try:
                notation = read_csa(csa_file)
            except ValueError:
                continue

            # 指定範囲以外のレーティングの場合は除外
            if not self.__min_rate <= max(notation.black.rate, notation.white.rate) <= self.__max_rate:
                continue

            # 投了や入玉宣言勝ち、千日手、最大手数制限以外で終局した場合は除外
            if notation.result == GameResult.UNKNOWN:
                continue

            # 評価値
            values = np.asarray([move.value for move in notation.moves], dtype=np.int32)
            black_values = values[::2]
            white_values = values[1::2]

            result = notation.result

            # 先手勝利
            if notation.result == GameResult.BLACK_WIN and not self.__is_output_black(black_values):
                continue

            # 後手勝利
            elif notation.result == GameResult.WHITE_WIN and not self.__is_output_white(white_values):
                continue

            # 千日手
            elif notation.result == GameResult.SENNICHITE and not self.__is_output_sennichite(values, black_values, white_values):
                continue

            # 最大手数制限
            elif notation.result == GameResult.MAX_MOVES:
                # 先手勝勢
                if self.__is_output_black(black_values):
                    result = GameResult.BLACK_WIN

                # 後手勝勢
                elif self.__is_output_white(white_values):
                    result = GameResult.WHITE_WIN

                # その他
                else:
                    continue

            board = shogi.Board()

            # 棋譜出力
            for i, move_data in enumerate(notation.moves):
                _, move = Parser.parse_move_str(move_data.move, board)
                board.push_usi(move)

                # 勝利（最大手数制限の場合は勝勢）側の棋譜のみを出力
                # 千日手の場合は両方
                x = 1 if i % 2 == 0 else -1
                if (result == GameResult.BLACK_WIN and x == 1) or (result == GameResult.WHITE_WIN and x == -1) or result == GameResult.SENNICHITE:
                    # 評価値上限の場合は、それ以降の棋譜を出力しない。
                    if move_data.value * x > self.__end_value:
                        break

                    position, game_ply = split_sfen(f'sfen {board.sfen()}')
                    sfens[position] = min(sfens.get(position, game_ply), game_ply)

        return [f'sfen {position} {game_ply}' for position, game_ply in sfens.items()]

    def __is_output_black(self, black_values) -> bool:
        # 全ての評価値が0の場合
        if np.all(black_values == 0):
            return False

        nonzero_black_values = black_values[black_values != 0]

        # 評価値0以外の指し手が手数の半分未満の場合
        if np.size(nonzero_black_values) < np.size(black_values) * 0.5:
            return False

        # 指定値より不利になった場合や投了時の評価値が指定値未満の場合
        if np.min(black_values) < self.__min_black_value or np.max(black_values) < self.__max_black_value:
            return False

        # 直前の評価値との差が指定値以上に不利になった場合
        diff = np.diff(nonzero_black_values)
        if np.min(diff) <= -self.__max_diff_value:
            return False

        return True

    def __is_output_white(self, white_values) -> bool:
        # 全ての評価値が0の場合
        if np.all(white_values == 0):
            return False

        nonzero_white_values = white_values[white_values != 0]

        # 評価値0以外の指し手が手数の半分未満の場合
        if np.size(nonzero_white_values) < np.size(white_values) * 0.5:
            return False

        # 指定値より不利になった場合や投了時の評価値が指定値未満の場合
        if np.max(white_values) > self.__max_white_value or np.min(white_values) > self.__min_white_value:
            return False

        # 直前の評価値との差が指定値以上に不利になった場合
        diff = np.diff(nonzero_white_values)
        if np.max(diff) >= self.__max_diff_value:
            return False

        return True

    def __is_output_sennichite(self, values, black_values, white_values) -> bool:
        # 全ての評価値が0の場合
        if np.all(values == 0):
            return False

        # 先手: 指定値より形勢に差がある場合（有利や不利になった場合）
        min_black_value: int = self.__min_black_value if self.__min_draw_value is None else self.__min_draw_value
        if np.min(black_values) < min_black_value or np.max(black_values) > self.__max_draw_value:
            return False

        # 後手: 指定値より形勢に差がある場合（有利や不利になった場合）
        max_white_value: int = self.__max_white_value if self.__min_draw_value is None else -self.__min_draw_value  # pylint: disable=invalid-unary-operand-type
        if np.max(white_values) > max_white_value or np.min(white_values) < -self.__max_draw_value:
            return False

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help='csaファイルのディレクトリ')
    parser.add_argument('-o', '--output', default='book.sfen', help='sfenの出力先')

    rate_group = parser.add_argument_group('レーティング')
    rate_group.add_argument('--min_rate', type=int, default=0, help='レーティング最小値')
    rate_group.add_argument('--max_rate', type=int, default=5000, help='レーティング最大値')

    value_group = parser.add_argument_group('評価値')
    value_group.add_argument('--min_black_value', type=int, default=-10, help='先手勝利時における、先手側から見た最小評価値')
    value_group.add_argument('--max_black_value', type=int, default=100000, help='先手勝利時における、先手側から見た最大評価値')
    value_group.add_argument('--min_white_value', type=int, default=-150, help='後手勝利時における、後手側から見た最小評価値')
    value_group.add_argument('--max_white_value', type=int, default=100000, help='後手勝利時における、後手側から見た最大評価値')
    value_group.add_argument('--draw_value', type=int, default=150, help='千日手における、手番側から見た最大評価値')
    value_group.add_argument('--end_value', type=int, default=0, help='勝利側から見た終局時の最小評価値')
    value_group.add_argument('--diff_value', type=int, default=100000, help='直前の評価値との差')

    args = parser.parse_args()
    csa_path = Path(args.input)
    output_path = Path(args.output)

    make = MakeSfen(csa_path)
    make.set_rate_limit(3800, 1000000)
    make.set_result_filter(GameResult.BLACK_WIN, -10, 2000)
    make.set_result_filter(GameResult.WHITE_WIN, -150, 2000)
    make.set_result_filter(GameResult.SENNICHITE, None, 150)
    make.set_diff_value_limit(300)
    make.set_value_limit(800)
    sfens: List[str] = make.run()

    # ファイル出力
    with output_path.open('w', encoding='utf_8') as f:
        for sfen in tqdm(sfens):
            f.write(f'{sfen}\n')