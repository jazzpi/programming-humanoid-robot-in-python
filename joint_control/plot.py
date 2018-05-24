#!/usr/bin/env python2

import argparse
import json
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from itertools import chain
from math import ceil
from _tkinter import TclError


def only_duplicates(it):
    '''`it` is a list of lists.
    Return only the elements that are in all lists.'''
    dups = it[0]
    for l in it[1:]:
        dups = filter(lambda i: i in l, dups)
    return dups


class Plotter(object):
    '''A class that handles plotting the joint angles and bezier handles.'''
    def __init__(self, subplots):
        '''Create a new Plotter.

        `subplots` is a list of dicts of the form
        `{'files': ['foo.json', 'bar.json'], 'initial': 'JointBaz'}` where
        the `initial` item is optional.'''
        self._init_axes(len(subplots))

        self.fig.canvas.mpl_connect('key_press_event', self._key_event)

        self._init_subplots(subplots)

        self.redraw()

    def _init_axes(self, n):
        rows = cols = 1
        if n > 1:
            cols = 2
            rows = int(ceil(n / 2.0))
        self.fig, axes = plt.subplots(rows, cols, squeeze=False)
        self.axes = list(chain.from_iterable(axes))  # Flatten

    def _init_subplots(self, subplots):
        '''Read all the files from the subplots and setup self.subplots.'''
        self._data = {}
        self.subplots = []
        for i, subplot in enumerate(subplots):
            sp = []
            for path in subplot['files']:
                if path not in self._data:
                    with open(path) as fh:
                        self._data[path] = json.load(fh)

                data = OrderedDict()
                for joint in sorted(self._data[path]['record']):
                    data[joint] = {
                        'record': self._data[path]['record'][joint],
                        'handles': self._data[path]['handles'][joint]
                    }

                sp.append({
                    'name': path.replace('.json', ''),
                    'data': data
                })

            joints = only_duplicates(map(lambda s: s['data'].keys(), sp))
            for s in sp:
                for joint in s['data']:
                    if joint not in joints:
                        del s['data'][joint]

            active = 0
            if 'initial' in subplot:
                active = joints.index(subplot['initial'])

            self.subplots.append({
                'data': sp,
                'axes': self.axes[i],
                'joints': joints,
                'active': active
            })

    def _key_event(self, event):
        '''Event handler for a key event (change shown plot).'''
        if event.key == 'left':
            self._add_active(-1)
        elif event.key == 'right':
            self._add_active(1)
        else:
            return
        self.redraw()

    def _add_active(self, amount):
        '''Advance `amount` in each subplot.'''
        for subplot in self.subplots:
            subplot['active'] += amount
            subplot['active'] %= len(subplot['joints'])

    def finish(self):
        '''Block in matplotlib until the figure window is closed.
        Should only be called once all plots are done.'''
        # sleep(5)
        plt.show()

    def redraw(self):
        for subplot in self.subplots:
            joint = subplot['joints'][subplot['active']]
            ax = subplot['axes']
            ax.clear()
            ax.set_title(joint)
            for sp in subplot['data']:
                self._plot(ax, sp, joint)
            ax.legend(loc='best')
        self.fig.canvas.draw()

    def _plot(self, ax, data, joint):
        '''Plot the data for a joint into a figure.'''
        record = data['data'][joint]['record']
        handles = data['data'][joint]['handles']

        ax.plot(record['x'], record['y'], label=data['name'])

        # Draw Bezier handles
        times, keys = handles
        centers = OrderedDict()
        prevs = OrderedDict()
        nexts = OrderedDict()
        for i in range(len(times)):
            time = times[i]
            key = keys[i]
            val = key[0]
            centers[time] = val
            if key[1][1] != 0:
                pt = time + key[1][1]
                pv = val + key[1][2]
                ax.plot([pt, time], [pv, val], c='green')
                prevs[pt] = pv
            if key[2][1] != 0:
                nt = time + key[2][1]
                nv = val + key[2][2]
                ax.plot([time, nt], [val, nv], c='red')
                nexts[nt] = nv
        ax.plot(centers.keys(), centers.values(), 'bo')
        ax.plot(prevs.keys(), prevs.values(), 'go')
        ax.plot(nexts.keys(), nexts.values(), 'ro')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='append', nargs='+',
                        help='Add a subplot from a list of files. You can also'
                             ' specify an initial joint by adding initial= '
                             'here.')
    args = parser.parse_args()
    subplots = []
    for subplot in args.s:
        sp = {}
        for i, f in enumerate(subplot):
            if f.startswith('initial='):
                sp['initial'] = f.replace('initial=', '')
                del subplot[i]

        sp['files'] = subplot
        subplots.append(sp)

    Plotter(subplots).finish()
