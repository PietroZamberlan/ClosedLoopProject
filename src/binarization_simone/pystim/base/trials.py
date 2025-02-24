import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from pystim.io.csv import load_file as load_csv_file


class TrialsSettings:

    column_names = ['condition_nb', 'start_display_nb', 'end_display_nb']

    @classmethod
    def load(cls, path):

        data_frame = load_csv_file(path, expected_columns=cls.column_names)

        return cls(data_frame)

    def __init__(self, data_frame):

        self._data_frame = data_frame

    def __len__(self):

        return self._data_frame.__len__()

    @property
    def condition_nbs_sequence(self):

        return self._data_frame['condition_nb']

    @property
    def start_display_nbs_sequence(self):

        return self._data_frame['start_display_nb']

    @property
    def end_display_nbs_sequence(self):

        return self._data_frame['end_display_nb']

    def are_temporally_disjoint(self):

        answer = True
        indices = self.start_display_nbs_sequence.argsort()
        for k in range(0, len(indices) - 1):
            i1 = indices[k]
            i2 = indices[k + 1]
            if self.end_display_nbs_sequence[i1] >= self.start_display_nbs_sequence[i2]:
                answer = False
                break

        return answer

    def print_summary(self):

        print("number of trials: {}".format(len(self)))

        counts = self.condition_nbs_sequence.value_counts()
        counts_counts = counts.value_counts()
        for count, counts_count in counts_counts.items():
            sub_counts = counts[counts == count]
            indices = np.array(sub_counts.index)
            repr_indices = []
            indices.sort()
            repr_indices.append('[')
            repr_indices.append(str(indices[0]))
            for k in range(1, len(indices)):
                if indices[k - 1] + 1 == indices[k]:
                    if k + 1 < len(indices) and indices[k] == indices[k + 1] - 1:
                        pass
                    else:
                        repr_indices.append('-')
                        repr_indices.append(str(indices[k]))
                else:
                    repr_indices.append(',')
                    repr_indices.append(' ')
                    repr_indices.append(str(indices[k]))
            repr_indices.append(']')
            repr_indices = ''.join(repr_indices)
            print("{} trials: {}".format(count, repr_indices))

        return

    def plot_summary(self, condition_nbs=None, mode='trial'):

        if condition_nbs is None:
            condition_nbs = np.unique(self.condition_nbs_sequence)
        else:
            condition_nbs = np.array(condition_nbs)

        if mode == 'display':

            fig, ax = plt.subplots()
            rectangles_dict = {}
            for condition_nb, start_display_nb, end_display_nb in zip(self.condition_nbs_sequence, self.start_display_nbs_sequence, self.end_display_nbs_sequence):
                if condition_nb not in condition_nbs:
                    continue
                xy = (start_display_nb, float(np.where(condition_nbs == condition_nb)[0][0]) - 0.5)
                width = (end_display_nb + 1) - start_display_nb
                height = 1.0
                if condition_nb not in rectangles_dict:
                    rectangles_dict[condition_nb] = []
                rectangles_dict[condition_nb].append(Rectangle(xy, width, height))
            for condition_nb, rectangles in rectangles_dict.items():
                color = 'C{}'.format(condition_nb % 10)
                pc = PatchCollection(rectangles, facecolor=color, edgecolor=color)
                ax.add_collection(pc)
            x_min = np.min(self.start_display_nbs_sequence)
            x_max = np.max(self.end_display_nbs_sequence + 1)
            ax.set_xlim(x_min, x_max)
            y_min = float(0) - 0.5
            y_max = float(np.max(self.condition_nbs_sequence) - 1) + 0.5
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("display")
            ax.set_ylabel("condition")
            fig.tight_layout()
            plt.show()

        elif mode == 'trial':

            fig, ax = plt.subplots()
            rectangles_dict = {}
            for trial_nb, condition_nb in enumerate(self.condition_nbs_sequence):
                if condition_nb not in condition_nbs:
                    continue
                xy = (float(trial_nb) - 0.5, float(np.where(condition_nbs == condition_nb)[0]) - 0.5)
                width = 1.0
                height = 1.0
                if condition_nb not in rectangles_dict:
                    rectangles_dict[condition_nb] = []
                rectangles_dict[condition_nb].append(Rectangle(xy, width, height))
            for k, (condition_nb, rectangles) in enumerate(rectangles_dict.items()):
                color = 'C{}'.format(k % 10)
                pc = PatchCollection(rectangles, facecolor=color, edgecolor=color)
                ax.add_collection(pc)
            x_min = float(0) - 0.5
            x_max = float(len(self.condition_nbs_sequence) - 1) + 0.5
            ax.set_xlim(x_min, x_max)
            y_min = float(0) - 0.5
            y_max = float(len(condition_nbs) - 1) + 0.5
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("trial")
            ax.set_ylabel("condition")
            fig.tight_layout()
            plt.show()

        else:

            raise ValueError("unknown mode value: {}".format(mode))

        return


load = TrialsSettings.load
