from itertools import zip_longest
from more_itertools import split_before
import pandas
import pathlib


def indexes(source, value):
    return [i for i, x in enumerate(source) if x.startswith(value)]


path = pathlib.Path('2020')
csa_files = path.glob('**/*.csa')

skip_rate = 3800

for csa_file in csa_files:
    print(csa_file)

    lines = csa_file.read_text().splitlines()

    # 対局者名
    black_name = lines[1]
    white_name = lines[2]

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
    if result != 'TORYO' and result != 'KACHI' and result != 'SENNICHITE' or result != summary[0].upper():
        continue

    # 指し手と消費時間、読み筋の組み合わせを取得
    # 読み筋がない場合はデフォルト値を入力しておく。
    # また、読み筋以外のコメントは削除する。
    move_list = list(split_before(lines[rate_index+1:result_index], lambda x: x.startswith('+') or x.startswith('-')))
    move_list = [[a + b for a, b in zip_longest(x, ['', '', "'* 0"], fillvalue='')] if len(x) < 3 else x[:3] for x in move_list]

    # 勝利側のみを抽出
    win = 'draw'
    if result == 'TORYO' or result == 'KACHI':
        black_summary = summary[1].split(' ')
        white_summary = summary[2].split(' ')
        if black_summary[1] == 'win':
            move_list = move_list[::2]
            win = 'black'
        elif white_summary[1] == 'win':
            move_list = move_list[1::2]
            win = 'white'
        else:
            # 勝敗不明なため除外
            continue

    # 指し手と評価値、読み筋の組み合わせを取得
    move_data = pandas.DataFrame(move_list, columns=['move', 'time', 'info'], )
    info = move_data['info'].str.lstrip("'* ").str.split(' ', 1, expand=True)
    info.rename(columns={0: 'value', 1: 'pv'}, inplace=True)
    info['value'] = info['value'].astype(int)
    move_data = pandas.concat([move_data['move'], info], axis=1)

    # 勝利側が評価値を出力していない場合は除外
    if (move_data['value'] == 0).all():
        continue

    # ワンサイドゲーム以外は除外
    # 1. 先手勝利：先手側の評価値が終局まで-10以上
    # 2. 後手勝利：後手側の評価値が終局まで150以下
    # 3. 引き分け：先手側の評価値が終局まで-10以上150以下かつ後手側の評価値が終局まで-150以上150以下
    values = move_data['value']
    if win == 'black' and (values < -10).any():
        continue
    if win == 'white' and (values > 150).any():
        continue
    if win == 'draw':
        black_values = values[::2]
        white_values = values[1::2]
        if ((black_values < -10) | (black_values > 150)).any() or ((white_values < -150) | (white_values > 150)).any():
            continue