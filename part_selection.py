import part_generation as pg
import numpy as np
import random
import matplotlib.pyplot as plt


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

    view_assignment = np.zeros([image_count, view_count], dtype=bool)

    model_errors = np.ndarray([image_count, 1])
    model_errors[:] = np.nan

    print "Working on class "

    for c in range(1):

        print c

        # class_tr_ID respect to label
        class_tr_ID = np.ones([1, 1], dtype=bool)

        if np.sum(class_tr_ID) < 1:
            continue

        best_obj_value = -np.inf

        for k in range(iterations):

            h, a, d, s, obj_value, err = do_build_part_models(parts, part_locs, class_tr_ID, no_selected_parts, no_visible_parts, view_count)

            if obj_value > best_obj_value:
                best_obj_value = obj_value
                best_h = h
                best_a = a
                best_d = d
                best_s = s
                best_err = err

        part_visibility[[x for x in np.where(class_tr_ID)[0]]] = best_h
        anchor_points[[x for x in np.where(class_tr_ID)[0]]] = best_a
        shift_vectors[c, :] = best_d
        view_assignment[[x for x in np.where(class_tr_ID)[0]]] = best_s
        model_errors[[x for x in np.where(class_tr_ID)[0]]] = best_err

    tr_ID = class_tr_ID
    # Inference for test images
    channel_ids = np.argsort(-np.nansum(part_visibility, 0))

    part_visibility[[x for x in np.where(~tr_ID)[0]]] = False
    selected_img = [x for x in np.where(~tr_ID)[0]]
    if selected_img:
        part_visibility[selected_img, channel_ids] = True
    part_visibility = np.where(part_visibility == 1, True, False)

    return channel_ids, part_visibility, parts


def do_build_part_models(parts, part_locs, class_tr_ID, no_selected_parts, no_visible_parts, view_count):

    part_ids = np.unique(parts['part'])
    image_ids = np.unique(parts['image'])
    part_count = len(part_ids)
    image_count = len(class_tr_ID)

    # part_locs = part_locs[:, class_tr_ID, :]

    # Variables to estimate
    # View selection for each image
    s = np.zeros([image_count, view_count], dtype=bool)

    # part selection b (indicator vetor) for each view
    b = np.zeros([view_count, part_count], dtype=bool)

    # Anchor points for each image
    a = np.zeros([image_count, 2])

    # Shift vectors for each part in each view
    d = np.zeros([part_count, view_count, 2])

    # Visibility of each part in each image
    h = np.zeros([image_count, part_count], dtype=bool)

    for i in range(image_count):
        s[i][np.random.randint(0, view_count)] = True

    for v in range(view_count):
        init_c = random.sample(range(part_count), no_selected_parts)
        for j in init_c:
            b[v][j] = True

    chose_visible_channel = np.where(parts['visible'] > 0)
    x = []
    y = []
    for c in chose_visible_channel[0]:
        x.append(parts['x'][c])
        y.append(parts['y'][c])
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    a = np.matlib.repmat([mean_x, mean_y], a.shape[0], 1)

    d = np.zeros(d.shape)

    if ~np.isnan(no_visible_parts):
        for i in range(image_count):
            available_parts = np.where(b[(np.where(s[i])[0]), :][0])[0]
            selected_visible_parts = [available_parts[x] for x in (random.sample(range(available_parts.size), no_visible_parts))]
            for j in selected_visible_parts:
                h[i][j] = True

    h = np.transpose(np.reshape(parts['visible'], [len(np.unique(parts['part'])), len(np.unique(parts['image']))]))
    h = np.where(h == 0, False, True)
    # there need tr_ID
    h = h[0:len(class_tr_ID), :]
    i = 0

    done = False
    best_obj_value = np.inf

    while ~done and np.ceil(np.float(i)/2) < 15:
        i = i + 1
        old_b = b

        # image_count * view_count * part_count * coordinates
        mu_tmp = np.transpose(np.expand_dims(part_locs, 3), [1, 3, 0, 2])
        a_tmp = np.transpose(np.expand_dims(np.expand_dims(a, 2), 3), [0, 2, 3, 1])
        d_tmp = np.transpose(np.expand_dims(d, 3), [3, 1, 0, 2])

        if i % 2 == 1:
            # Estimate d
            mu_a = mu_tmp - a_tmp
            mu_a = np.tile(mu_a, [1, view_count, 1, 1])

            mask = np.ones([image_count, view_count, part_count], dtype=bool)
            mask = mask & np.transpose(np.expand_dims(h, 2), [0, 2, 1])
            mask = np.tile(np.expand_dims(mask, 3), [1, 1, 1, 2])

            mu_a = np.multiply(mu_a, np.where(mask==True, 1, 0))
            mu_a = np.where(mu_a != 0, mu_a, np.nan)

            d = np.nanmean(mu_a, 0)
            d = np.transpose(d, [1, 0, 2])

            # Estimate a
            mu_d = mu_tmp - d_tmp
            mu_d = np.multiply(mu_d, np.where(mask==True, 1, 0))
            mu_d = np.where(mu_d != 0, mu_d, np.nan)

            a = np.nanmean(np.nanmean(mu_d, 2), 1)

        else:
            mu_a_d = mu_tmp - (a_tmp + d_tmp)
            mu_a_d = np.sum(mu_a_d**2, 3)

            # Estimate h
            if ~np.isnan(no_visible_parts):
                est_h = np.zeros(h.shape, dtype=bool)

                mask = np.ones([image_count, view_count, part_count], dtype=bool)
                mask = mask & np.transpose(np.expand_dims(s, 2), [0, 1, 2])
                mask = mask & np.transpose(np.expand_dims(b, 2), [2, 0, 1])

                mu_a_d_tmp = mu_a_d
                mu_a_d_tmp = np.multiply(mu_a_d_tmp, np.where(mask == True, 1, 0))
                mu_a_d_tmp = np.where(mu_a_d_tmp != 0, mu_a_d_tmp, np.nan)
                mu_a_d_tmp = np.nansum(mu_a_d_tmp, 1)
                mu_a_d_tmp = np.expand_dims(mu_a_d_tmp, 1)

                mu_a_d_tmp = np.where(mu_a_d_tmp == 0, np.inf, mu_a_d_tmp)

                idx = np.argsort(mu_a_d_tmp, 2)
                idx = np.transpose(idx, [1, 0, 2])[0]
                idx = idx[:, 0:no_visible_parts]
                idx2 = np.tile(np.expand_dims(np.arange(idx.shape[0]), 1), [1, idx.shape[1]])
                idx2_tmp = np.transpose(idx2).flatten()
                idx_tmp = np.transpose(idx).flatten()
                for coor in range(len(idx2_tmp)):
                    est_h[idx2_tmp[coor]][idx_tmp[coor]] = True

            # Estimate s
            s = np.zeros(s.shape, dtype=bool)

            mask = np.ones([image_count, view_count, part_count], dtype=bool)
            mask = mask & np.transpose(np.expand_dims(h, 2), [0, 2, 1])
            mask = mask & np.transpose(np.expand_dims(b, 2), [2, 0, 1])

            mu_a_d_tmp = mu_a_d
            mu_a_d_tmp = np.multiply(mu_a_d_tmp, np.where(mask == True, 1, 0))
            mu_a_d_tmp = np.where(mu_a_d_tmp != 0, mu_a_d_tmp, np.nan)
            mu_a_d_tmp = np.nansum(mu_a_d_tmp, 2)

            idx = np.argsort(mu_a_d_tmp, 1)
            idx = idx[:, 0:1]
            idx2 = np.tile(np.expand_dims(np.arange(idx.shape[0]), 1), [1, idx.shape[1]])
            idx2_tmp = np.transpose(idx2).flatten()
            idx_tmp = np.transpose(idx).flatten()
            for coor in range(len(idx2_tmp)):
                s[idx2_tmp[coor]][idx_tmp[coor]] = True

            # Estimate b
            b = np.zeros(b.shape, dtype=bool)

            mask = np.ones([image_count, view_count, part_count], dtype=bool)
            mask = mask & np.expand_dims(s, 2)

            mu_a_d_tmp = mu_a_d
            mu_a_d_tmp = np.multiply(mu_a_d_tmp, np.where(mask == True, 1, 0))
            mu_a_d_tmp = np.where(mu_a_d_tmp != 0, mu_a_d_tmp, np.nan)
            mu_a_d_tmp = np.nansum(mu_a_d_tmp, 0)

            idx = np.argsort(mu_a_d_tmp, 1)
            idx = idx[:, 0:no_selected_parts]
            idx2 = np.tile(np.expand_dims(np.arange(idx.shape[0]), 1), [1, idx.shape[1]])
            idx2_tmp = np.transpose(idx2).flatten()
            idx_tmp = np.transpose(idx).flatten()
            for coor in range(len(idx2_tmp)):
                b[idx2_tmp[coor]][idx_tmp[coor]] = True

            # Remember old b to check for convergence
            if (old_b == b).all():
                done = True
            else:
                old_b = b

    # Get the error of each training image
    mu_tmp = np.transpose(np.expand_dims(part_locs, 3), [1, 3, 0, 2])
    a_tmp = np.transpose(np.expand_dims(np.expand_dims(a, 2), 3), [0, 2, 3, 1])
    d_tmp = np.transpose(np.expand_dims(d, 3), [3, 1, 0, 2])

    mu_a_d = mu_tmp - (a_tmp + d_tmp)
    mu_a_d = np.sum(mu_a_d**2, 3)

    mask = np.ones([image_count, view_count, part_count], dtype=bool)
    mask = mask & np.expand_dims(s, 2)
    mask = mask & np.transpose(np.expand_dims(h, 2), [0, 2, 1])
    mask = mask & np.transpose(np.expand_dims(b, 2), [2, 0, 1])

    mu_a_d = np.multiply(mu_a_d, np.where(mask == True, 1, 0))
    mu_a_d = np.where(mu_a_d != 0, mu_a_d, np.nan)

    err = np.nansum(np.nansum(mu_a_d, 1), 1)
    obj_value = -np.nansum(err)

    # Return values
    part_visibility = est_h
    anchor_points = a
    shift_vectors = d
    view_assignment = s

    return part_visibility, anchor_points, shift_vectors, view_assignment, obj_value, err


def show_keypoint( part_visibility, parts, channel_ids):

    part_count = 435

    CONTENT_IMG = '/home/alala/Projects/part_constellation_models_tensorflow/images/Taipei101.jpg'

    key_points = []

    # show the keypoint
    for i in range(len(part_visibility)):
        #selected_channel_ids = np.where(part_visibility[i, :])
        selected_channel_ids = channel_ids
        selection = parts['visible'][np.add(selected_channel_ids, i * part_count)]
        cur_channels = selected_channel_ids[[x for x in np.where(selection)]]
        x = parts['x'][i * part_count + cur_channels]
        y = parts['y'][i * part_count + cur_channels]

        coors = np.dstack((x, y))

        im = plt.imread(CONTENT_IMG)
        plt.figure(), plt.imshow(im)

        for j in coors[0]:
            info = j

            x = info[0]

            y = info[1]

            plt.plot(x, y, 'rx')

        plt.show()

        key_points.append(coors)

    return key_points


if __name__ == '__main__':

    part_locs = pg.part_generation()

    channel_ids, part_visibility, parts = part_selection(part_locs, view_count=10, iterations=5, no_selected_parts=10, no_visible_parts=5)

    show_keypoint(part_visibility, parts, channel_ids[0:10])

    print 's'


