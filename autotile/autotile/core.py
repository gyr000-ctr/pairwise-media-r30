"""autotile 核心 API：47-tile 自动拼接的掩码、变体表与 atlas 索引。"""

import math
import threading
from collections import deque, namedtuple

__all__ = [
    "AutoTile",
    "AutoTileError",
    "VariantsError",
    "GridError",
    "BoundsError",
    "NonFiniteError",
    "MaskError",
    "Fallback",
    "TableEntry",
    "NORTH",
    "EAST",
    "SOUTH",
    "WEST",
    "NORTH_EAST",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "NORTH_WEST",
    "CARDINAL_MASK",
    "DIRECTION_BITS",
    "OPPOSITE_BITS",
    "CORNER_SUPPORTS",
    "VARIANT_TABLE",
    "TABLE_ENTRIES",
    "INDEX_OF_MASK",
    "ATLAS_COLUMNS",
    "TABLE_SIZE",
    "normalize_mask",
    "raw_mask8",
    "variant_index_of",
    "atlas_cell_of",
    "atlas_mask_of",
    "reciprocal_ok",
]

# 8 位掩码的位序：四个正邻接在低 4 位，四个对角在高 4 位。
NORTH = N = 1
EAST = E = 2
SOUTH = S = 4
WEST = W = 8
NORTH_EAST = NE = 16
SOUTH_EAST = SE = 32
SOUTH_WEST = SW = 64
NORTH_WEST = NW = 128

CARDINAL_MASK = NORTH | EAST | SOUTH | WEST

# 47-tile 的规范掩码只有 47 个；变体表与查表都以这个数量为准。
TABLE_SIZE = 47

# atlas 的默认宽度；47 个变体按行优先排入，最后一格留空。
ATLAS_COLUMNS = 8

DIRECTION_BITS = {
    "n": NORTH,
    "e": EAST,
    "s": SOUTH,
    "w": WEST,
    "ne": NORTH_EAST,
    "se": SOUTH_EAST,
    "sw": SOUTH_WEST,
    "nw": NORTH_WEST,
}

# 相反方向的位：判定相邻块互逆时使用。
OPPOSITE_BITS = {
    NORTH: SOUTH,
    EAST: WEST,
    SOUTH: NORTH,
    WEST: EAST,
    NORTH_EAST: SOUTH_WEST,
    SOUTH_EAST: NORTH_WEST,
    SOUTH_WEST: NORTH_EAST,
    NORTH_WEST: SOUTH_EAST,
}

# 每个对角位依赖的两个正邻接位。
CORNER_SUPPORTS = (
    (NORTH_EAST, NORTH, EAST),
    (SOUTH_EAST, SOUTH, EAST),
    (SOUTH_WEST, SOUTH, WEST),
    (NORTH_WEST, NORTH, WEST),
)

_isfinite = math.isfinite

Fallback = namedtuple("Fallback", "x y mask8 variant index")
TableEntry = namedtuple("TableEntry", "index mask row col")


class AutoTileError(ValueError):
    """自动拼接错误的基类，继承 ValueError 让调用方按值错误处理。"""


class VariantsError(AutoTileError):
    """变体数量、默认变体、atlas 列数或回退记录容量不合法。"""


class GridError(AutoTileError):
    """网格结构不合法：类型不对、为空、行不是序列或各行长度不一致。"""


class BoundsError(AutoTileError):
    """坐标不是整数或超出网格范围。"""


class NonFiniteError(AutoTileError):
    """网格里出现 NaN 或无穷值。"""


class MaskError(AutoTileError):
    """掩码不是 0 到 255 之间的整数。"""


def normalize_mask(mask):
    """把任意 8 位掩码规整成 47 项变体表里的规范掩码。

    对角位描述两块正邻接之间的缺角：相邻的正邻接有一侧为空时，
    该对角位不参与变体选择。孤立角（只有对角、两侧正邻接都缺）与对角缺口
    （一侧正邻接缺）都靠这条规则归入已有变体，不会落到回退分支。
    """
    if type(mask) is not int or not 0 <= mask <= 255:
        raise MaskError("掩码必须是 0 到 255 之间的整数")
    for corner, side_a, side_b in CORNER_SUPPORTS:
        if not (mask & side_a and mask & side_b):
            mask &= ~corner
    return mask


def _build_variant_table():
    """按掩码升序枚举 256 个掩码的规范形式，得到 47 项变体表。

    枚举顺序固定，索引就是该规范掩码在升序表中的位置。前 16 项恰好是
    不带对角位的四邻接组合，所以 variants=16 时可用变体就是这 16 种；
    索引 0 是八邻接全空的孤立块，索引 46 是八邻接全满的块。
    """
    canonical = [mask for mask in range(256) if normalize_mask(mask) == mask]
    if len(canonical) != TABLE_SIZE:
        raise RuntimeError(f"47-tile 规范掩码数量应为 {TABLE_SIZE}，实际为 {len(canonical)}")
    return tuple(canonical)


VARIANT_TABLE = _build_variant_table()

_INDEX_OF_CANONICAL = [0] * 256
for _index, _mask in enumerate(VARIANT_TABLE):
    _INDEX_OF_CANONICAL[_mask] = _index

# 任意 8 位掩码直接查索引，热路径只做一次元组索引。
INDEX_OF_MASK = tuple(_INDEX_OF_CANONICAL[normalize_mask(mask)] for mask in range(256))

# 变体表与 atlas 的对照：每一项都带自己的行列，atlas 以行优先排列。
TABLE_ENTRIES = tuple(
    TableEntry(index, mask, index // ATLAS_COLUMNS, index % ATLAS_COLUMNS)
    for index, mask in enumerate(VARIANT_TABLE)
)


def _check_mask(mask):
    if type(mask) is not int or not 0 <= mask <= 255:
        raise MaskError("掩码必须是 0 到 255 之间的整数")


def variant_index_of(mask):
    """8 位掩码到 47 项变体表的索引。"""
    _check_mask(mask)
    return INDEX_OF_MASK[mask]


def atlas_cell_of(mask, columns=ATLAS_COLUMNS):
    """由掩码得到 atlas 的行列索引；atlas 以行优先存放 47 个变体。"""
    if type(columns) is not int or columns < 1:
        raise VariantsError("atlas 列数必须是正整数")
    return divmod(variant_index_of(mask), columns)


def atlas_mask_of(row, col, columns=ATLAS_COLUMNS):
    """由 atlas 行列还原规范掩码，与 atlas_cell_of 互逆。"""
    if type(columns) is not int or columns < 1:
        raise VariantsError("atlas 列数必须是正整数")
    if type(row) is not int or type(col) is not int or row < 0 or col < 0:
        raise BoundsError("atlas 行列必须是非负整数")
    index = row * columns + col
    if index >= TABLE_SIZE:
        raise BoundsError("atlas 行列超出变体表范围")
    return VARIANT_TABLE[index]


def reciprocal_ok(mask_a, mask_b, direction):
    """相邻两块在给定方向上是否互逆一致。

    A 认为 B 在 direction 一侧时，B 必须在相反方向认为 A 存在。
    """
    bit = DIRECTION_BITS.get(direction)
    if bit is None:
        raise ValueError(f"未知方向：{direction}")
    return bool(mask_a & bit) == bool(mask_b & OPPOSITE_BITS[bit])


def _filled(value):
    """判断单元格是否填充；非有限值与非数值在此分类报错。"""
    kind = type(value)
    if kind is int:
        return value != 0
    if kind is float:
        if not _isfinite(value):
            raise NonFiniteError("网格中出现非有限值")
        return value != 0.0
    if kind is bool:
        return value
    raise GridError(f"网格元素必须是 int 或 float，实际为 {kind.__name__}")


def _row(grid, y, width):
    """取一行并校验类型与宽度。"""
    row = grid[y]
    kind = type(row)
    if kind is not list and kind is not tuple:
        raise GridError("网格的每一行必须是 list 或 tuple")
    if len(row) != width:
        raise GridError("网格各行的长度必须一致")
    return row


def _check_grid(grid, x, y):
    """校验网格结构与坐标，返回 (本行, 宽, 高)。"""
    grid_kind = type(grid)
    if grid_kind is not list and grid_kind is not tuple:
        raise GridError("网格必须是 list 或 tuple")
    height = len(grid)
    if height == 0:
        raise GridError("网格不能为空")
    if type(x) is not int or type(y) is not int:
        raise BoundsError("坐标必须是 int")
    if not 0 <= y < height:
        raise BoundsError(f"坐标越界：({x}, {y}) 不在网格内")
    head = grid[y]
    kind = type(head)
    if kind is not list and kind is not tuple:
        raise GridError("网格的每一行必须是 list 或 tuple")
    width = len(head)
    if width == 0:
        raise GridError("网格的每一行不能为空")
    if not 0 <= x < width:
        raise BoundsError(f"坐标越界：({x}, {y}) 不在网格内")
    return head, width, height


def raw_mask8(grid, x, y):
    """计算 (x, y) 的原始 8 位掩码，同时校验网格与坐标。

    空格返回 0：掩码描述一块填充瓦片四周的拼接关系，空位没有变体。
    越界一格折回对侧边缘（宽或高为 1 时该方向没有邻居），因此贴边的填充
    区域不会被地图边界切断，任意两块相邻填充块的同一方向位严格互逆。
    只读取本次邻域涉及的 3 行与 9 个单元格，单次查询是常数时间。
    """
    row, width, height = _check_grid(grid, x, y)

    has_cols = width > 1
    has_rows = height > 1

    if has_rows:
        up_y = y - 1 if y > 0 else height - 1
        down_y = y + 1 if y + 1 < height else 0
        up_row = _row(grid, up_y, width)
        down_row = _row(grid, down_y, width)
    else:
        up_row = down_row = None

    if not _filled(row[x]):
        return 0

    mask = 0
    if has_rows:
        if _filled(up_row[x]):
            mask |= NORTH
        if _filled(down_row[x]):
            mask |= SOUTH
    if has_cols:
        xl = x - 1 if x > 0 else width - 1
        xr = x + 1 if x + 1 < width else 0
        if _filled(row[xl]):
            mask |= WEST
        if _filled(row[xr]):
            mask |= EAST
        if has_rows:
            if _filled(up_row[xl]):
                mask |= NORTH_WEST
            if _filled(up_row[xr]):
                mask |= NORTH_EAST
            if _filled(down_row[xl]):
                mask |= SOUTH_WEST
            if _filled(down_row[xr]):
                mask |= SOUTH_EAST
    return mask


def _verify_tables():
    if len(VARIANT_TABLE) != TABLE_SIZE:
        raise RuntimeError("变体表项数错误")
    if len(set(VARIANT_TABLE)) != TABLE_SIZE:
        raise RuntimeError("变体表存在重复掩码")
    if list(VARIANT_TABLE) != sorted(VARIANT_TABLE):
        raise RuntimeError("变体表必须按掩码升序排列")
    if VARIANT_TABLE[0] != 0 or VARIANT_TABLE[-1] != 255:
        raise RuntimeError("变体表首尾应为全空与全满")
    for entry in TABLE_ENTRIES:
        if normalize_mask(entry.mask) != entry.mask:
            raise RuntimeError("变体表包含非规范掩码")
        if entry.row * ATLAS_COLUMNS + entry.col != entry.index:
            raise RuntimeError("atlas 行列与变体索引不一致")
        if atlas_mask_of(entry.row, entry.col) != entry.mask:
            raise RuntimeError("atlas 行列还原的掩码与变体表不一致")
    for index, mask in enumerate(VARIANT_TABLE):
        if INDEX_OF_MASK[mask] != index:
            raise RuntimeError("掩码索引与变体表不一致")
        if atlas_cell_of(mask) != (TABLE_ENTRIES[index].row, TABLE_ENTRIES[index].col):
            raise RuntimeError("掩码生成的 atlas 行列与变体表不一致")
    for raw in range(256):
        canonical = normalize_mask(raw)
        if VARIANT_TABLE[INDEX_OF_MASK[raw]] != canonical:
            raise RuntimeError("任意掩码查表结果与规范掩码不一致")
    for columns in (1, 2, 3, 7, 8, 16, 47, 64):
        for index, mask in enumerate(VARIANT_TABLE):
            row, col = divmod(index, columns)
            if atlas_cell_of(mask, columns) != (row, col):
                raise RuntimeError("非默认列数的 atlas 行列与变体表不一致")
            if atlas_mask_of(row, col, columns) != mask:
                raise RuntimeError("非默认列数的 atlas 还原与变体表不一致")


_verify_tables()


class AutoTile:
    """47-tile 自动拼接：由邻接网格选出 atlas 变体索引。

    variants 是 atlas 中实际可用的变体数量；规范掩码对应的索引超出该数量时，
    回退到 default_variant，并记录一次缺失。
    """

    TABLE = VARIANT_TABLE
    ENTRIES = TABLE_ENTRIES
    TABLE_SIZE = TABLE_SIZE

    def __init__(self, variants, default_variant=0, atlas_columns=ATLAS_COLUMNS, log_limit=64):
        if type(variants) is float:
            if not _isfinite(variants) or not variants.is_integer():
                raise VariantsError("变体数量必须是有限整数")
            variants = int(variants)
        elif type(variants) is not int:
            raise VariantsError("变体数量必须是整数")
        if variants < 1:
            raise VariantsError("变体数量必须大于 0")

        if type(default_variant) is not int:
            raise VariantsError("默认变体必须是整数")
        if not 0 <= default_variant < min(variants, TABLE_SIZE):
            raise VariantsError("默认变体必须在可用变体范围内")

        if type(atlas_columns) is not int or atlas_columns < 1:
            raise VariantsError("atlas 列数必须是正整数")
        if type(log_limit) is not int or log_limit < 1:
            raise VariantsError("回退记录容量必须是正整数")

        self._variants = variants
        self._limit = min(variants, TABLE_SIZE)
        self._default = default_variant
        self._columns = atlas_columns
        self._count = 0
        self._recent = deque(maxlen=log_limit)
        self._lock = threading.Lock()

    @property
    def variants(self):
        return self._variants

    @property
    def default_variant(self):
        return self._default

    @property
    def atlas_columns(self):
        return self._columns

    @property
    def atlas_shape(self):
        """atlas 的行数与列数。"""
        rows = (TABLE_SIZE + self._columns - 1) // self._columns
        return rows, self._columns

    @property
    def table(self):
        return VARIANT_TABLE

    @property
    def entries(self):
        return TABLE_ENTRIES

    @property
    def fallback_count(self):
        """累计回退次数。"""
        return self._count

    @property
    def fallbacks(self):
        """最近若干次回退记录，元素为 Fallback(x, y, mask8, variant, index)。"""
        with self._lock:
            return tuple(self._recent)

    @property
    def last_fallback(self):
        with self._lock:
            return self._recent[-1] if self._recent else None

    def clear_fallbacks(self):
        """清空回退计数与记录。"""
        with self._lock:
            self._count = 0
            self._recent.clear()

    def mask(self, grid, x, y):
        """四邻接掩码（N=1, E=2, S=4, W=8），取值 0 到 15。"""
        return raw_mask8(grid, x, y) & CARDINAL_MASK

    def mask8(self, grid, x, y):
        """原始 8 位掩码（四邻接 + 四对角），取值 0 到 255。"""
        return raw_mask8(grid, x, y)

    def variant_index(self, mask):
        """8 位掩码对应的 47 项变体表索引。"""
        _check_mask(mask)
        return INDEX_OF_MASK[mask]

    def atlas_cell(self, mask):
        """8 位掩码对应的 atlas 行列索引。"""
        _check_mask(mask)
        return divmod(INDEX_OF_MASK[mask], self._columns)

    def mask_from_atlas(self, row, col):
        """由 atlas 行列还原规范掩码，与 atlas_cell 互逆。"""
        return atlas_mask_of(row, col, self._columns)

    def pick(self, grid, x, y):
        """按八位掩码选择变体；可用变体不足时回退到默认变体并记录。"""
        mask8 = raw_mask8(grid, x, y)
        index = INDEX_OF_MASK[mask8]
        if index < self._limit:
            return index
        record = Fallback(x, y, mask8, VARIANT_TABLE[index], index)
        with self._lock:
            self._count += 1
            self._recent.append(record)
        return self._default
