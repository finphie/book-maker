from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BookPos:
    best_move: str
    next_move: str
    value: int
    depth: int
    num: int

    def __str__(self) -> str:
        return ' '.join((self.best_move, self.next_move, str(self.value), str(self.depth), str(self.num)))


@dataclass(frozen=True)
class Book:
    game_ply: int
    body: List[BookPos]

    def __str__(self) -> str:
        return '\n'.join(str(x) for x in self.body)


class BookFormatter:
    @staticmethod
    def deserialize(buffer: List[str]) -> Book:
        game_ply = int(buffer[0].rsplit(' ', 1)[1])
        body: List[BookPos] = []

        for x in buffer[1:]:
            a = x.split(' ')
            book_pos = BookPos(best_move=a[0], next_move=a[1], value=int(a[2]), depth=int(a[3]), num=int(a[4]))
            body.append(book_pos)

        return Book(game_ply=game_ply, body=body)

    @staticmethod
    def serialize(book: Book) -> List[str]:
        result = [BookFormatter.__version]
        for x in book.body:
            result.append(str(x))

        return result