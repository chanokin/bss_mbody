from __future__ import (print_function,
                        # unicode_literals,
                        division)
from builtins import open, range, dict

import numpy as np
import copy
import os
from .wafer_blacklists import BLACKLISTS

def flip_coords(row_coords):
    coords = {}
    for row in row_coords:
        for col in row_coords[row]:
            col_list = coords.get(col, [])
            col_list.append(row)
            coords[col] = col_list

    for col in coords:
        coords[col] = np.array(sorted(coords[col]))

    return coords

def compute_coords(row_widths, row_starts):
    coords = {}
    i2c = {}
    c2i = {}
    id_start = 0
    for i in range(len(row_widths)):
        id_start += 0 if i == 0 else row_widths[i - 1]
        width = row_widths[i]
        start = row_starts[i]
        coords[i] = start + np.arange(width)
        ids = id_start + np.arange(width)
        i2c.update({ids[j]: (i, coords[i][j]) for j in range(width)})
        c2i[i] = {coords[i][j]: ids[j] for j in range(width)}

    return coords, i2c, c2i


def compute_row_id_starts(row_widths):
    starts = np.roll(np.cumsum(row_widths), 1)
    starts[0] = 0
    starts.flags.writeable = False
    return starts


def compute_row_ranges(row_widths):
    '''
    per row = (starting id [inclusive], ending id [exclusive])
    so we can do range[0] <= x < range[1] which should help with slices
    '''
    ranges = []
    s = 0
    for w in row_widths:
        ranges.append((s, s + w))
        s += w
    return tuple(map(tuple, ranges))


WAFER_COL_HEIGHTS = np.array([4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16,
                      16, 16, 16, 16, 16, 16, 16, 16, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4])
WAFER_COL_HEIGHTS.flags.writeable = False
WAFER_MIN_HEIGHT = 4
WAFER_ROW_WIDTHS = np.array([12, 12, 20, 20, 28, 28, 36, 36, 36, 36, 28, 28, 20, 20, 12, 12])
WAFER_ROW_WIDTHS.flags.writeable = False
WAFER_MIN_WIDTH = 12
WAFER_ROW_X_STARTS = np.array([12, 12, 8, 8, 4, 4, 0, 0, 0, 0, 4, 4, 8, 8, 12, 12])
WAFER_ROW_X_STARTS.flags.writeable = False
WAFER_COL_Y_STARTS = np.array([6, 6, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6])
WAFER_COL_Y_STARTS.flags.writeable = False
WAFER_ROW_ID_STARTS = compute_row_id_starts(WAFER_ROW_WIDTHS)
WAFER_ROW_RANGES = compute_row_ranges(WAFER_ROW_WIDTHS)
WAFER_ROW_COORDS, WAFER_I2C, WAFER_C2I = compute_coords(WAFER_ROW_WIDTHS, WAFER_ROW_X_STARTS)
WAFER_COL_COORDS = flip_coords(WAFER_ROW_COORDS)


WAFER_CHIPS_PER_RETICLE = 8
WAFER_CHIPS_SHAPE = (2, 4)
WAFER_RETICLE_ROW_WIDTHS = np.array([3, 5, 7, 9, 9, 7, 5, 3])
WAFER_RETICLE_ROW_WIDTHS.flags.writeable = False
WAFER_RETICLE_ROW_X_STARTS = np.array([3, 2, 1, 0, 0, 1, 2, 3])
WAFER_RETICLE_ROW_X_STARTS.flags.writeable = False


class Wafer(object):
    _col_heights = WAFER_COL_HEIGHTS
    _min_height = WAFER_MIN_HEIGHT
    _row_widths = WAFER_ROW_WIDTHS
    _min_width = WAFER_MIN_WIDTH
    _row_x_starts = WAFER_ROW_X_STARTS
    _col_y_starts = WAFER_COL_Y_STARTS
    _row_id_starts = WAFER_ROW_ID_STARTS
    _row_ranges = WAFER_ROW_RANGES
    _row_coords = WAFER_ROW_COORDS
    _col_coords = WAFER_COL_COORDS
    _i2c = WAFER_I2C
    _c2i = WAFER_C2I
    _width = 36  ### max(row_widths)
    _height = 16  ### rows per wafer
    _reticle_row_widths = WAFER_RETICLE_ROW_WIDTHS
    _reticle_row_x_starts = WAFER_RETICLE_ROW_X_STARTS

    def __init__(self, wafer_id=33):
        self._id = wafer_id
        self._avail_ids = None
        self._used_chips = [{} for _ in range(self._height)]

        self._blacklist = BLACKLISTS.get(self._id, {})

        self._get_distances()

    def __iter__(self):
        if self._avail_ids is None:
            yield None
        else:
            for i in range(len(self._avail_ids)):
                yield i

    def _get_distances(self):
        cwd = os.path.dirname( os.path.realpath(__file__) )
        dist_file = os.path.join(cwd, 'wafer_distance_cache.npz')
        # if os.path.isfile(dist_file):
        #     # print(np.load(dist_file).keys())
        #     loaded = np.load(dist_file)
        #     self.distances = loaded['distances']
        #     self.id2idx = loaded['id2idx'].item()
        #     return

        id2idx = {}
        ids, coords = self.available()
        # n_avail = len(ids)
        # distances = np.zeros((n_avail, n_avail))
        distances = {}
        for i in ids:
            # id2idx[ids[i]] = i
            ri, ci = coords[i]
            dists = distances.get(i, dict())
            for j in ids:
                if i == j:
                    dists[j] = 10.0**3 #big number to avoid NaN later
                else:
                    rj, cj = coords[j]
                    dists[j] = np.sqrt((ri - rj)**2 + (ci - cj)**2)
                    # dists[j] = np.abs(ri - rj) + np.abs(ci - cj)
            distances[i] = dists
        np.savez_compressed(dist_file, distances=distances, id2idx=id2idx)
        self.distances = distances
        self.id2idx = id2idx


    def available(self, width=None, height=None, max_id=np.inf):
        if width is None or height is None:
            max_rows = self._height
            max_cols = self._width
            orientation = 'vertical'
            start_y = 0
            start_x = 0
        else:
            max_size = (width if width > height else height)
            if max_size >= self._height or max_size >= self._width:
                max_rows = self._height
                max_cols = self._width
                orientation = 'vertical'
                start_y = 0
                start_x = 0
            else:

                start_x = self._row_x_starts[0]
                start_y = self._col_y_starts[0]
                try:
                    col_idx = np.where(max_size > self._col_heights)[0][0]
                except:
                    col_idx = np.inf
                try:
                    row_idx = np.where(max_size > self._row_widths)[0][0]
                except:
                    row_idx = np.inf
                orientation = 'vertical' if row_idx > col_idx else 'horizontal'
                max_rows = max_size
                max_cols = max_size

        ids = []
        coords = {}
        if orientation == 'vertical':
            end_x = start_x + max_cols
            for row in range(max_rows):
                mask = np.where(np.logical_and(start_x <= self._row_coords[row],
                                               self._row_coords[row] < end_x))
                columns = sorted(self._row_coords[row][mask])
                for col in columns:
                    _id = self._c2i[row][col]
                    if _id > max_id:
                        continue
                    
                    bl = self._blacklist.get(row, [])
                    if _id not in self._used_chips[row] and _id not in bl:
                        ids.append(_id)
                        coords[_id] = (row, col)
        else:
            end_y = start_y + max_rows
            for col in range(max_cols):
                mask = np.where(np.logical_and(start_y <= self._col_coords[col],
                                               self._col_coords[col] < end_y))
                rows = sorted(self._col_coords[col][mask])
                for row in rows:
                    _id = self._c2i[row][col]
                    if _id > max_id:
                        continue
                    
                    bl = self._blacklist.get(row, [])
                    if _id not in self._used_chips[row] and _id not in bl:
                        ids.append(_id)
                        coords[_id] = (row, col)
        self._avail_ids = ids
        return ids, coords

    def _in_range(self, chip_id, id_range):
        return (id_range[0] <= chip_id < id_range[1])

    def find_row(self, chip_id):
        '''Find wafer row given a chip id. To locate the row, since we don't
        have a nice rectangular grid, we need to locate when the difference
        between row id start and the chip id changes from negative to
        positive or it is zero.
        '''
        diffs = np.array(self._row_id_starts) - chip_id
        whr = np.where(diffs <= 0)[0]
        ### last element is always the closest to 0 (inclusive) thus
        ### the correct row
        return whr[-1]

    def insert(self, pop_id, chip_id):
        row = self.find_row(chip_id)
        if chip_id in self._used_chips[row]:
            print('This chip was alread used, try another one')
            return False
        if chip_id in self._blacklist[row]:
            print('This chip was blacklisted, try another one')
            return False

        self._used_chips[row][chip_id] = pop_id
        return True

    def remove(self, chip_id):
        row = self.find_row(chip_id)
        try:
            del self._used_chips[row][chip_id]
            return True
        except:
            print('Chip ({}) not not found in register'.format(chip_id))
            return False

    def render(self, fig_width=15):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        fig = plt.figure(figsize=(fig_width, fig_width))

        ax = plt.subplot(1, 1, 1)

        for row in self._c2i:
            for col in self._c2i[row]:
                _id = self._c2i[row][col]
                if _id in self._used_chips[row]:
                    color = 'blue'
                elif _id in self._blacklist[row]:
                    color = 'red'
                else:
                    color = 'green'

                plt.text(col, row, '{:3d}'.format(_id),
                         horizontalalignment='center', verticalalignment='center',
                         bbox=dict(  # xy=(col-0.5, row-0.5), width=1., height=1.,
                             facecolor=color, edgecolor='none', alpha=0.333)
                         )

        ax.set_xlim(-1, self._width + 1)
        ax.set_ylim(self._height + 1, -1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.tight_layout()
        plt.savefig('wafer_render.pdf')
        plt.close(fig)
        print('Render saved to {}'.format(os.path.join(os.getcwd(), 'wafer_render.pdf')))


    def clone(self):
        wfr = Wafer(self._id)
        wfr._used_chips[:] = [copy.copy(used) for used in self._used_chips]
        return wfr

    def i2c(self, i):
        return self._i2c[i]

    def c2i(self, coord_or_row, col=None):
        if col is None:
            row = coord_or_row[0]
            col = coord_or_row[1]
        else:
            row = coord_or_row

        return self._c2i[row][col]


    def get_neighbours(self, center_id, max_dist=1):
        """:param center_id: the id of the hicann from whom neighbours are required
        :param max_dist: how wide the neighbourhood is"""
        if max_dist == 0:
            raise  Exception("Unable to provide neighbourhood for a distance of 0")

        data = {}
        c_row, c_col = self.i2c(center_id)
        data[0] = {0: [center_id, c_row, c_col]}
        
        for dr in np.arange(-max_dist, max_dist + 1):
            row_dict = data.get(dr, dict())
            for dc in np.arange(-max_dist, max_dist + 1):
                if dr == 0 and dc == 0:
                    continue

                r = c_row + dr
                if r not in self._row_coords:
                    continue

                c = c_col + dc
                if c not in self._row_coords[r]:
                    continue

                nid = self.c2i([r, c])

                row_dict[dc] = [nid, r, c]

            data[dr] = row_dict

        return data


