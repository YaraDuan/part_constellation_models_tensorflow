import part_generation as pg
import numpy as np


def part_selection(part_locs, view_count, iterations, no_selected_parts, no_visible_parts):

    parts = {}
    parts['image'] = part_locs[:, 0]
    parts['part'] = part_locs[:, 1]
    parts['x'] = part_locs[:, 2]
    parts['y'] = part_locs[:, 3]
    parts['visible'] = part_locs[:, 4]

    part_ids = np.unique(parts['part'])
    part_count = len(part_ids)

    part_x = np.reshape(parts['x'], [len(np.unique(parts['part'])), len(np.unique(parts['image']))])
    part_y = np.reshape(parts['y'], [len(np.unique(parts['part'])), len(np.unique(parts['image']))])

    part_locs = np.dstack((part_x, part_y))
    part_locs = part_locs[0:part_count, :]

    image_ids = np.unique(parts['image'])
    image_count = len(image_ids)

    # initialization
    part_visibility = np.ndarray([image_count, part_count])
    part_visibility[:] = np.nan

    anchor_points = np.ndarray([image_count, 2])
    anchor_points[:] = np.nan

    # 1 == label's number
    shift_vectors = np.ndarray([1, part_count, view_count, 2])
    shift_vectors[:] = np.nan

    view_assignment = np.ndarray([image_count, view_count])
    view_assignment[:] = False

    model_errors = np.ndarray([image_count, 1])
    model_errors[:] = np.nan

    for c in range(1):

        # class_tr_ID respect to label
        class_tr_ID = [[1]]

        if np.sum(class_tr_ID) < 1:
            continue

        best_obj_value = -np.inf

        for k in range(iterations):

            h, a, d ,s , obj_value, err = do_build_part_models(parts, part_locs, class_tr_ID, no_selected_parts, no_visible_parts, view_count)

            if obj_value > best_obj_value:
                best_obj_value = obj_value
                best_h = h
                best_a = a
                best_d = d
                best_s = s
                best_err = err

        part_visibility[class_tr_ID, :] = best_h
        anchor_points[class_tr_ID, :] = best_a
        shift_vectors[c+1, :] = best_d
        view_assignment[class_tr_ID, :] = best_s
        model_errors[class_tr_ID, :] = best_err




def do_build_part_models(parts, part_locs, class_tr_ID, no_selected_parts, no_visible_parts, view_count):

    part_ids = np.unique(parts['part'])
    image_ids = np.unique(parts['image'])
    part_count = len(part_ids)
    image_count = len(class_tr_ID)

    # part_locs = part_locs[:, class_tr_ID, :]

    # Variables to estimate
    # View selection for each image
    s = np.zeros([image_count, view_count])

    # part selection b (indicator vetor) for each view
    b = np.zeros([view_count, part_count])

    # Anchor points for each image
    a = np.zeros([image_count, 2])

    # Shift vectors for each part in each view
    d = np.zeros([part_count, view_count, 2])

    # Visibility of each part in each image
    h = np.zeros([image_count, part_count])

    for i in range(image_count):
        s[i][np.random.randint(0, view_count)] = True

    for v in range(view_count):
        init_c = np.random.randint(0, part_count, no_selected_parts)
        for j in init_c:
            b[v][j] = 1

    chose_visible_channel = np.where(parts['visible'] > 0)
    x = []
    y = []
    for c in chose_visible_channel[0]:
        x.append(parts['x'][c])
        y.append(parts['y'][c])
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    a = np.matlib.repmat([mean_x, mean_y], a.shape[0], 1)

    if ~np.isnan(no_visible_parts):
        for i in range(image_count):
            print 'h'

    h = []
    a = []
    d = []
    s = []
    obj_value = []
    err = []

    return h, a, d ,s , obj_value, err


if __name__ == '__main__':

    part_locs = pg.part_generation()

    part_selection(part_locs, view_count=10, iterations=5, no_selected_parts=10, no_visible_parts=5)