from more_itertools import split_before
import pandas
import pathlib
import shogi
import shogi.CSA


def indexes(source, value):
    return [i for i, x in enumerate(source) if x.startswith(value)]


path = pathlib.Path('2020')
csa_files = path.glob('**/*.csa')

skip_rate = 3800
sfen = []

for csa_file in csa_files:
    print(csa_file)

    notation = shogi.CSA.Parser.parse_file(csa_file)[0]
    lines = csa_file.read_text().splitlines()

    # 対局者名
    black_name = notation['names'][shogi.BLACK]
    white_name = notation['names'][shogi.WHITE]

    # レーティング
    rate_index = lines.index('+') + 2
    black_rate = white_rate = 0.0
    if lines[rate_index].startswith("'black_rate:"):
        black_rate = float(lines[rate_index].split(':')[-1])
        rate_index += 1
    if lines[rate_index].startswith("'white_rate:"):
        white_rate = float(lines[rate_index].split(':')[-1])

    # 指定値未満のレーティングの場合は除外
    if skip_rate > max(black_rate, white_rate):
        continue

    # 終局結果を取得
    if not (result_indexes := indexes(lines, '%')):
        continue
    result_index = result_indexes[0]
    result = lines[result_index][1:]
    summary = lines[indexes(lines, "'summary:")[0]].split(':')[1:]

    # 投了と入玉宣言勝ち、千日手以外で終局した場合は除外
    # summaryをチェックしているのは、%TORYOが送信された場合でもabnormalになっていることがあるため。
    if result not in ['TORYO', 'KACHI', 'SENNICHITE'] or result != summary[0].upper():
        continue

    # 評価値と読み筋の組み合わせを取得
    move_list = list(split_before(lines[rate_index+1:result_index], lambda x: x.startswith('+') or x.startswith('-')))
    move_list = [x[2].lstrip("'* ").split(' ', 1) if len(x) > 2 else [0, None] for x in move_list]

    # 指し手と評価値、読み筋の組み合わせを取得
    move_data = pandas.concat([pandas.Series(notation['moves'], name='move'), pandas.DataFrame(move_list, columns=['value', 'pv'])], axis=1)
    move_data['value'] = move_data['value'].astype(int)

    black_move_data = move_data[::2]
    white_move_data = move_data[1::2]

    values = move_data['value']
    black_values = black_move_data['value'].copy()
    white_values = white_move_data['value'].copy()

    # 先手勝利
    if notation['win'] == 'b':
        # 1. 先手側が評価値を出力していない場合は除外
        # 2. 先手側の評価値が終局まで-10以上
        if (black_values == 0).all() or (black_values < -10).any():
            continue

        # 評価値0は定跡などの要因で出力される場合があるので無視する。
        black_values[black_values == 0] = None
        black_values = black_values.fillna(method='bfill')

        # 直前の評価値から300以上下がった場合は除外
        if (black_values.diff() <= -300).any():
            continue
 
        # sfen出力
        board = shogi.Board()
        for data in black_move_data.itertuples():
            # 評価値1000より大きい場合は、それ以降の指し手を記録しない。
            if data.value > 1000:
                break

            board.push_usi(data.move)
            sfen.append(board.sfen())
            board.push_usi(move_data['move'][data.Index + 1])

    # 後手勝利
    elif notation['win'] == 'w':
        # 1. 後手側が評価値を出力していない場合は除外
        # 2. 後手側の評価値が終局まで150以下
        if (white_values == 0).all() or (white_values > 150).any():
            continue

        # 評価値0は定跡などの要因で出力される場合があるので無視する。
        white_values[white_values == 0] = None
        white_values = white_values.fillna(method='bfill')

        # 直前の評価値から300以上下がった場合は除外
        if (white_values.diff() >= 300).any():
            continue

        # sfen出力
        board = shogi.Board()
        for data in white_move_data.itertuples():
            # 評価値-1000より小さい場合は、それ以降の指し手を記録しない。
            if data.value < -1000:
                break

            board.push_usi(move_data['move'][data.Index - 1])
            board.push_usi(data.move)
            sfen.append(board.sfen())

    # 引き分け
    elif notation['win'] == '-':
        # 先後両方が評価値を出力していない場合は除外
        if (values == 0).all():
            continue

        # 先手側の評価値が終局まで-10以上150以下かつ後手側の評価値が終局まで-150以上150以下
        if ((black_values < -10) | (black_values > 150)).any() or ((white_values < -150) | (white_values > 150)).any():
            continue

        # sfen出力
        board = shogi.Board()
        for data in move_data.itertuples():
            board.push_usi(data.move)
            sfen.append(board.sfen())