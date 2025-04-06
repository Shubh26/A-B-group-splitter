from z3 import *
set_option(verbose=0)
set_param('parallel.enable', True)
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import control_test_splitter as split_utils
from collections import defaultdict
import json, datetime, itertools, os, pickle
import pandas as pd, numpy as np
from scipy.spatial import ConvexHull
from geopy import distance
from sklearn.metrics import pairwise_distances
ASSETS_DIR_RELPATH = 'assets'

import matplotlib
matplotlib.use('Agg') #added by VG 9/1/23 https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc

def load_zipcodes():
    """
    This produces a dict of zipcodes to lat/long coordinates.
    :return:
    """
    assets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ASSETS_DIR_RELPATH)
    zipcode_to_latlong = {}
    with open(os.path.join(assets_dir, 'US.txt')) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line == "":
                continue
            tokens = line.split('\t')
            # print(len(tokens))
            if len(tokens) == 12:
                zipcode, lat_x, long_y = tokens[1], float(tokens[-3]), float(tokens[-2])
            else:
                zipcode, lat_x, long_y = tokens[1], float(tokens[-2]), float(tokens[-1])
            zipcode_to_latlong[zipcode.strip()] = (float(lat_x), float(long_y))

    return zipcode_to_latlong


def plot_store_sets(store_splits, store_centroids, plot_file, control_key='control'):
    """
    Visualizes store allocations to different subsets. Control is (a shade of) red. Other colors denote different test
    sets. This is an approximate viz. - on a @D cartesian plane instead of a geographic map.
    :param store_splits: dict of store split across subsets, as might be returned by set_splitter()
    :param store_centroids: dict of stores mapped to their centroid lat/long tuple.
    :param plot_file: output file for visualization
    :param control_key: key in store_splits that denotes the control subset. NOTE: if store_splits has no 'control',
        this still works, and will plot other sets as test sets.
    :return:
    """
    # function level constants for color, style etc
    facealpha, edgealpha = 0.5, 1.0
    control_color = [0.74609375, 0.12890625, 0.12890625]
    test_colors = [[0., 0.30, 0.35], [0.0, 0.43, 0.42], [0.92, 0.85, 0.61], [0.96, 0.55, 0.28], [0.94, 0.43, 0.24]]

    # TODO: we should check for tests and control separately, which the error must reflect.
    if len(store_splits) > 6:
        print("Can plot up to 6 sets only, you have provided %d. Aborting." % (len(store_splits),))
        return

    test_sizes, control_size = [], 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    test_idx = -1
    for idx, (subset_name, stores) in enumerate(store_splits.items()):
        temp = np.array([store_centroids[s] for s in stores])
        if subset_name == control_key:
            ax.scatter(temp[:, 0], temp[:, 1], facecolor=control_color+[facealpha],
                        edgecolor=control_color+[edgealpha],)
            control_size = len(stores)
        else:
            test_idx += 1
            ax.scatter(temp[:, 0], temp[:, 1], facecolor=test_colors[test_idx]+[facealpha],
                        edgecolor=test_colors[test_idx]+[edgealpha])
            test_sizes.append(len(stores))
    plt.title("control=%d, test=%s" % (control_size, ",".join(map(str, test_sizes))))
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()


def expand_groups_into_stores(subsets_with_name, group_name_to_store_map):
    """
    If the set splitter is called with a grouping of stores, then the final result can be used only after expanding the
    groups back into the constituent stores. This function performs the expansion.
    :param subsets_with_name: as returned by the set_splitter() function, or something similar.
    :param group_name_to_store_map: dict of group names mapped to their constituent list of stores, possibly returned by
        group_stores()
    :return:
    """
    store_splits = {}
    for subset_name, group_list in subsets_with_name.items():
        temp = [group_name_to_store_map[i] for i in group_list]
        store_splits[subset_name] = flatten_2d_lists(temp)
    return store_splits


def get_geo_dist_matrix(X):
    """
    Compute a pairwise distance matrix with geodesic distances.
    :param X:
    :return:
    """
    def geo_miles(i, j):
        return distance.distance(i, j).miles
    print("Begining pairwise dist calculations ...this will take a while.")
    D = pairwise_distances(X, metric=geo_miles)
    return D


def flatten_2d_lists(d):
    return list(itertools.chain.from_iterable(d))


def group_stores(data_json, zip_to_latlong_map, n_groups=10, distance_metric="euclidean",
                 plot_file=None):
    """
    This functions groups stores based on some heuristics, typically clustering.
    This is the naming convention: what we want is to *group* stores, and *one way* to do it via clustering.
    :param data_json: the retail data dictionary
    :param zip_to_latlong_map: a dictionary mapping a zipcode to a 2-tuple of lat, long coordinates.
        NOTE: if there are zipcodes in the data_json that are absent here, processing doesn't stop; it prints the
        missing zipcodes and works with what is available.
    :param n_groups: the number of groups we want; note, this is a hard partition, i.e., the same store cannot belong
        to two groups
    :param distance_metric: since we are clustering now, the distance metric to use. Can be 'euclidean' or
        'geo'/'geodesic'. The latter is the right way to measure distances on the surface of the Earth, and the geopy
        package is used, which internally uses the Vincenty distance on the WGS84 geodetic model. The Euclidean option
        allows for a fast approximation.
        Accuracy-wise: a.euclidean < b.haversine < c.vincenty on WGS84 < d.vincenty on NAD83 (for N. America). But we
        can only use a or c. Geopy doesn't support NAD83.
    :param plot_file: the clusters can be plotted out to this file. This is a simplified visualization where stores are
        denoted by centroids computed from the associated zipcodes, and the groups are visualized using convex hulls.
        Note the centroid computation is Euclidean, which is a good approximation, since the associated zipcodes
        are supposed to be close-by.
    :return: dict of cluster names to list of stores in the cluster
    :return: dict of stores to 2D centroid coordinates.
    """
    stores_all_zips = {}
    for store, info in data_json.items():
        stores_all_zips[store] = set(info['control_influence']).union(set(info['test_influence']))
    valid_zipcodes = set.union(*[set(v) for k, v in stores_all_zips.items()])
    t = valid_zipcodes - set(zip_to_latlong_map.keys())
    if len(t) == 0:
        print("Found latlong for all zipcodes.")
    else:
        raise Exception("%d zipcodes missing in latlong map!!!: %s" % (len(t), ",".join(t)))

    centroids = []
    store_to_centroid = {}
    for store_idx, (store, zipcodes) in enumerate(stores_all_zips.items()):
        temp = [zip_to_latlong_map[z] for z in set(zipcodes) if z in zip_to_latlong_map]
        temp = np.array(temp)
        # this is technically inaccurate since we should use a geodesic distance, but since this is guaranteed to be
        # a local neighborhood (region around a store), this is likely to have very low error.
        centroid = np.mean(temp, axis=0)
        centroids.append(centroid.tolist())
        store_to_centroid[store] = centroid.tolist()
    centroids = np.array(centroids)

    # cluster
    ordered_stores = sorted(stores_all_zips.keys())
    num_stores = len(ordered_stores)
    store_to_id_map = dict([(j, i) for i, j in enumerate(ordered_stores)])

    # dist_mat = np.ones((num_stores, num_stores), dtype=float)
    X = [store_to_centroid[s] for s in ordered_stores]
    if distance_metric == "euclidean":
        cluster_labels = AgglomerativeClustering(n_clusters=n_groups, affinity="euclidean",
                                                 linkage="average").fit(X).labels_
    elif distance_metric in ['geo', 'geodesic']:
        dist_mat = get_geo_dist_matrix(X)
        cluster_labels = AgglomerativeClustering(n_clusters=n_groups, affinity="precomputed",
                                                 linkage="average").fit(dist_mat).labels_
    else:
        print("Can't understand metric: %s" % (distance_metric,))

    print("Requested %d clusters, found %d clusters." % (n_groups, len(set(cluster_labels))))
    groupings = defaultdict(list)
    for store_id, cluster_label in enumerate(cluster_labels):
        groupings[cluster_label].append(ordered_stores[store_id])

    # draw a hull around the clusters
    if plot_file:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(centroids[:, 0], centroids[:, 1], alpha=0.5)

        for stores in groupings.values():
            cluster_zips = []
            for store in stores:
                for z in stores_all_zips[store]:
                    if z in zip_to_latlong_map:
                        cluster_zips.append(zip_to_latlong_map[z])
            cluster_zips = np.array(cluster_zips)

            if len(stores) < 3:
                # cant plot a convex hull with less than 3 points - plot a line instead
                ax.plot(cluster_zips[:, 0], cluster_zips[:, 1], c='k', marker='o', markerfacecolor='none')
                continue

            hull = ConvexHull(cluster_zips, qhull_options='QJ')
            vertices_cycle = hull.vertices.tolist()
            vertices_cycle.append(hull.vertices[0])
            ax.plot(cluster_zips[vertices_cycle, 0], cluster_zips[vertices_cycle, 1], 'k-', lw=1)

        ax.set_title("# stores=%d, # groups=%d" % (num_stores, n_groups))
        plt.savefig(plot_file, bbox_inches='tight')
    return groupings, store_to_centroid


def merge_stores(data_json, groupings):
    """
    Create a new data json file, with the size attribute, that reflects the clustering obtained.
    :param data_json:
    :param groupings:
    :return:
    """
    num_stores, num_clusters = len(data_json), len(groupings)
    print("Merging %d store data into %d store clusters." % (num_stores, num_clusters))

    store_to_cluster_name_map = dict([(store_name, "cluster_%s" % (str(k))) for k, v in groupings.items()
                                      for store_name in v])
    cluster_name_to_store_map = defaultdict(list)
    for k, v in store_to_cluster_name_map.items():
        cluster_name_to_store_map[v].append(k)
    # cluster_name_to_store_map = dict([(v, k) for k, v in store_to_cluster_name_map.items()])
    merged_data = defaultdict(lambda: {'control_influence': [], 'test_influence': [], 'score': 0, 'size': 0})
    for store, store_info in data_json.items():
        cluster_name = store_to_cluster_name_map[store]
        merged_data[cluster_name]['test_influence'] = list(set(merged_data[cluster_name]['test_influence'] +
                                                               store_info['test_influence']))
        merged_data[cluster_name]['control_influence'] = list(set(merged_data[cluster_name]['control_influence'] +
                                                               store_info['control_influence']))
        merged_data[cluster_name]['score'] += store_info['score']
        merged_data[cluster_name]['size'] += 1  # size of this cluster

    return merged_data, cluster_name_to_store_map


def count_val_bool(vec, val):
    if val == 0:
        return Sum([If(i, 0, 1) for i in vec])
    else:
        return Sum([If(i, 1, 0) for i in vec])


def elem_sum_bool(alloc_vec, cost_vec, alloc_id):
    if alloc_id == 0:
        return Sum([j * If(i, 0, 1) for i, j in zip(alloc_vec, cost_vec)])
    else:
        return Sum([j * If(i, 1, 0) for i, j in zip(alloc_vec, cost_vec)])


def find_conflicting_zipcodes(retail_data):
    """
    Given store data figure what which zipcodes occur in the control of one store and test of a different store and
    vice-versa.
    :param retail_data:
    :return:
    """
    print("Beginning analysis for conflicting zipcodes.")
    zipcodes_influence = defaultdict(lambda: defaultdict(set))
    subkeys = ['control_influence', 'test_influence']
    for store, info in retail_data.items():
        for sk in subkeys:
            for z in info[sk]:
                zipcodes_influence[z][sk].add(store)
    print("Total zipcodes: %d" % (len(zipcodes_influence)))
    conflict_zipcodes = []
    for z, h in zipcodes_influence.items():

        if len(h['control_influence']) == 0 or len(h['test_influence'])==0:
            # if the zip occurs as test or control influence ONLY, across stores, it cannot create conflict
            continue
        elif len(h['control_influence']) == 1 and h['control_influence'] == h['test_influence']:
            # no conflict if the control and test sets, created across stores, contain exactly one store
            # and its the same store
            continue
        else:
            # everything else is a potential conflict
            conflict_zipcodes.append(z)
    print("Conflicted zipcodes: %d (%0.02f%%)" %
          (len(conflict_zipcodes), 100.0 * len(conflict_zipcodes) / len(zipcodes_influence) ))
    return set(conflict_zipcodes)


def set_splitter_binary(data_json, splits, results_file, avg_tol=5, size_tol=5, use_soft_constraints=False,
                 solver_obj=None, only_return_solver=False, suffix_str=None, solve_as_SAT=False, SAT_upper_bound=None,
                 use_only_conflicted_zipcodes=True):
    """
    Like set_splitter(), but set_allocations are binary: to check if this saves us time.
    :return:
    """
    assert 'control' in splits
    assert sum(splits.values()) == 1
    assert(len(splits) == 2)
    if suffix_str is None:
        suffix_str = ""

    ordered_stores = sorted(data_json.keys())
    # print("Ordered stores: %s" % (", ".join(ordered_stores)))
    ordered_costs = [data_json[i]['score'] for i in ordered_stores]

    # get IDs for set names
    set_name_to_id = {'control': 0}
    set_id_count = 0
    for k in splits:
        if k != 'control':
            set_id_count += 1
            set_name_to_id[k] = set_id_count
    print("set name to ID map (for debugging): %s" % (str(set_name_to_id)))
    set_id_to_name = dict([(v, k) for k, v in set_name_to_id.items()])

    # store id costs

    # get and print basic stats
    num_sets, num_elems = len(splits), len(data_json)
    print("Avg score tolerance: %0.02f" % (avg_tol,))
    print("Size tolerance: %d elements" % (size_tol,))
    print("Num sets: %d, Total elements (stores): %d" % (num_sets, num_elems))

    # we adopt the convention that sets are denoted by integers >= 0, where 0 always indicates control group
    # set_alloc = IntVector('set_alloc'+suffix_str, num_elems)
    set_alloc = [Bool('set_alloc_%d'%(i+1,))for i in range(num_elems)]
    # reuse a solver object if one is passed in
    # also check if we want to solve as SAT
    if solver_obj:
        s = solver_obj
    elif solve_as_SAT:
        s = Solver()
    else:
        s = Optimize()

    # sets would be identified by set IDs
    # for i in range(num_elems):
    #     s.add(And(set_alloc[i] > -1, set_alloc[i] < num_sets))

    # Create some common collections - for setting constraints later
    # This probably could be replaced with a native type, but to be safe, I am letting this stay,
    # till we have a test setup.
    expected_subset_sizes = RealVector('expected_subset_sizes'+suffix_str, num_sets)
    current_subset_sizes = IntVector('current_subset_sizes'+suffix_str, num_sets)
    current_subset_scores = RealVector('current_subset_scores'+suffix_str, num_sets)
    for i in range(num_sets):
        s.add(expected_subset_sizes[i] == splits[set_id_to_name[i]] * num_elems)
        s.add(current_subset_sizes[i] == count_val_bool(set_alloc, i))
        s.add(current_subset_scores[i] == elem_sum_bool(set_alloc, ordered_costs, i))

    # size constraints - compare expected size to actual size
    s.add(And([split_utils.z3_abs(current_subset_sizes[i] - expected_subset_sizes[i]) <= size_tol
               for i in range(num_sets)]))

    # subset average score constraints - averages must much to a tolerance
    control_avg_score = Real('control_avg_score'+suffix_str)
    s.add(control_avg_score == current_subset_scores[0]/current_subset_sizes[0])
    s.add(And([split_utils.z3_abs(current_subset_scores[i] / current_subset_sizes[i] - control_avg_score) <= avg_tol
     for i in range(1, num_sets)]))

    # store the number of constraints a zipcode can have; we need this because z3 vecs are need a length spec
    influence_constraints_zipcode = defaultdict(lambda: defaultdict(list))  # count per store per influence category
    for store_idx, store in enumerate(ordered_stores):
        for zipcode in data_json[store]['control_influence']:
            influence_constraints_zipcode[zipcode]['control_influence'].append((store_idx, store))
        for zipcode in data_json[store]['test_influence']:
            influence_constraints_zipcode[zipcode]['test_influence'].append((store_idx, store))

    if use_only_conflicted_zipcodes:
        conflicted_zipcodes = find_conflicting_zipcodes(data_json)
    else:
        conflicted_zipcodes = set(influence_constraints_zipcode.keys())

    overlap_vec = []
    z3_zipcode_constraint_vecs = {}
    for zipcode_idx, (zipcode, h) in enumerate(influence_constraints_zipcode.items()):
        if zipcode not in conflicted_zipcodes:
            continue
        control_OR = Or([Not(set_alloc[store_idx]) for store_idx, store in h['control_influence']])
        test_OR = Or([set_alloc[store_idx] for store_idx, store in h['test_influence']])

        temp_overlap = Bool('zipcode_overlap_%d' % (zipcode_idx))
        s.add(temp_overlap == (And(control_OR, test_OR)))
        overlap_vec.append(temp_overlap)

    num_overlaps = Real("num_overlaps"+suffix_str)
    s.add(num_overlaps>=0)
    s.add(num_overlaps==Sum([If(i, 1, 0) for i in overlap_vec]))

    # return the solver object, without solving SAT, if requested
    if only_return_solver:
        return s, num_overlaps, z3_zipcode_constraint_vecs, set_alloc, ordered_stores, set_id_to_name

    print("Number of assertions created: %d" %(len(s.assertions())))
    # print(s.simplify())
    start_time = datetime.datetime.now()
    print("Beginning solution at %s." % (str(start_time)))

    if solve_as_SAT:
        s.add(num_overlaps <= SAT_upper_bound)
    else:
        s.minimize(num_overlaps)



    if s.check() == sat:
        finish_time = datetime.datetime.now()
        duration_sec = (finish_time - start_time).total_seconds()
        print("Found solution! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
        m = s.model()
        # print(m)
        subsets = defaultdict(list)
        subsets_with_name = defaultdict(list)
        for s_idx, store in enumerate(ordered_stores):
            subset_id = BoolVal(m[set_alloc[s_idx]])
            subset_id = 1 if subset_id else 0
            subsets[subset_id].append(store)
            subsets_with_name[set_id_to_name[subset_id]].append(store)
        split_utils.write_results_to_df(data_json, subsets_with_name, results_file,
                                        other_info="Runtime: %f sec" % (duration_sec,))
        # for k, v in sorted(subsets.items()):
        #     subset_size = len(v)
        #     subset_type = 'control' if k==0 else 'test'
        #     print("subset_%s, size=%d, %s: %s" % (k, subset_size, subset_type, ",".join(v)))
        return subsets_with_name
    elif s.check() == unsat:
        print("unsatisfiable condition")
    elif s.check() == unknown:
        print("unknown end condition")
    else:
        finish_time = datetime.datetime.now()
        duration_sec = (finish_time - start_time).total_seconds()
        print("No solution found. (at=%s, total=%f sec)" % (str(finish_time), duration_sec))


def set_splitter(data_json, splits, results_file, avg_tol=5, size_tol=5, use_soft_constraints=False,
                 solver_obj=None, only_return_solver=False, suffix_str=None, solve_as_SAT=False, SAT_upper_bound=None,
                 use_only_conflicted_zipcodes=True, use_size_info=False, use_only_combined_conflicted_zipcode = False, combined_conflicted_zipcodes = None):
    """
    Assumption: one control group, possibly more than one test group
    A lot of operations are now moved to Boolean conditions so that z3 has a better chance to optimize stuff. Also this
    takes up less memory than having Ints.
    :param data_json:
    :param splits: a dict with entries for split fractions of subsets. It is mandatory to have one key as "control"
    :param results_file: this is where results would be written as csv
    :param avg_tol: how much can the subset scores averages differ by
    :param size_tol: how much can subset sizes differ by, beyond what is expected from the splits
    :param solver_obj: pass in an existing solver object to which constraints are to be added.
    :param only_return_solver: don't solve, and importantly do not add the minimization constraint - only return solver
    :param suffix_str: if not None or greater than zero length, is appended to zipcode var names inside z3 - this is
        to avoid naming conflicts in case caller needs to separate out same zipcodes across calls in the same solver obj
    :param solve_as_SAT: if you do not want to solve this as an optimize problem but a SAT problem. This might be
                helpful if some margin of error is acceptable on the zip overlaps, and you want to exit when you are
                below this. Also z3's SAT is supposedly more efficient than its optimizer.
    :param SAT_upper_bound: this variable is only considered if solve_as_SAT==True. This is the upper bound on the
                overlaps that decides the problem has found a SAT solution.
    :param use_only_conflicted_zipcodes: only zipcodes that can potentially create conflicts are included in constraints
                        this can save a lot of compute time, esp when used with grouping
    :param use_size_info: if this is True the input must have an additional field call size per store, and this is what
        we would use for matching subset sizes. If False (default), each store has a size of 1. This option is useful
        if stores have been clustered previously and we need to account for the # stores inside a cluster.
    :return:
    """
    assert 'control' in splits
    assert sum(splits.values()) == 1
    if suffix_str is None:
        suffix_str = ""

    ordered_stores = sorted(data_json.keys())
    # print("Ordered stores: %s" % (", ".join(ordered_stores)))
    ordered_costs = [data_json[i]['score'] for i in ordered_stores]
    ordered_sizes = None
    if use_size_info:
        ordered_sizes = [data_json[i]['size'] for i in ordered_stores]


    # get IDs for set names
    set_name_to_id = {'control': 0}
    set_id_count = 0
    for k in splits:
        if k != 'control':
            set_id_count += 1
            set_name_to_id[k] = set_id_count
    print("set name to ID map (for debugging): %s" % (str(set_name_to_id)))
    set_id_to_name = dict([(v, k) for k, v in set_name_to_id.items()])

    # store id costs

    # get and print basic stats
    num_sets, num_elems = len(splits), len(data_json)
    print("Avg score tolerance: %0.02f" % (avg_tol,))
    print("Size tolerance: %d elements" % (size_tol,))
    print("Num sets: %d, Total elements (stores): %d" % (num_sets, num_elems))

    # we adopt the convention that sets are denoted by integers >= 0, where 0 always indicates control group
    set_alloc = IntVector('set_alloc'+suffix_str, num_elems)

    # reuse a solver object if one is passed in
    # also check if we want to solve as SAT
    if solver_obj:
        s = solver_obj
    elif solve_as_SAT:
        s = Solver()
    else:
        s = Optimize()

    # sets would be identified by set IDs
    for i in range(num_elems):
        s.add(And(set_alloc[i] > -1, set_alloc[i] < num_sets))

    # Create some common collections - for setting constraints later
    # This probably could be replaced with a native type, but to be safe, I am letting this stay,
    # till we have a test setup.
    expected_subset_sizes = RealVector('expected_subset_sizes'+suffix_str, num_sets)
    current_subset_sizes = IntVector('current_subset_sizes'+suffix_str, num_sets)
    current_subset_scores = RealVector('current_subset_scores'+suffix_str, num_sets)
    for i in range(num_sets):
        # add size info constraint based on the use_size_info constraint
        if use_size_info:
            num_stores = sum(ordered_sizes)
            s.add(expected_subset_sizes[i] == splits[set_id_to_name[i]] * num_stores)
            s.add(current_subset_sizes[i] == split_utils.elem_sum(set_alloc, ordered_sizes, num_elems, i))
        else:
            s.add(expected_subset_sizes[i] == splits[set_id_to_name[i]] * num_elems)
            s.add(current_subset_sizes[i] == split_utils.count_val(set_alloc, num_elems, i))

        s.add(current_subset_scores[i] == split_utils.elem_sum(set_alloc, ordered_costs, num_elems, i))

    # size constraints - compare expected size to actual size
    s.add(And([split_utils.z3_abs(current_subset_sizes[i] - expected_subset_sizes[i]) <= size_tol
               for i in range(num_sets)]))

    # subset average score constraints - averages must much to a tolerance
    control_avg_score = Real('control_avg_score'+suffix_str)
    s.add(control_avg_score == current_subset_scores[0]/current_subset_sizes[0])
    s.add(And([split_utils.z3_abs(current_subset_scores[i] / current_subset_sizes[i] - control_avg_score) <= avg_tol
     for i in range(1, num_sets)]))

    # store the number of constraints a zipcode can have; we need this because z3 vecs are need a length spec
    influence_constraints_zipcode = defaultdict(lambda: defaultdict(list))  # count per store per influence category
    for store_idx, store in enumerate(ordered_stores):
        for zipcode in data_json[store]['control_influence']:
            influence_constraints_zipcode[zipcode]['control_influence'].append((store_idx, store))
        for zipcode in data_json[store].get('test_influence',[]):
            influence_constraints_zipcode[zipcode]['test_influence'].append((store_idx, store))

    if use_only_conflicted_zipcodes:
        conflicted_zipcodes = find_conflicting_zipcodes(data_json)
        # if there are no constraints there is nothing to optimize, only SAT. Not sure how z3 handles this.
        # TODO: what is the correct way to handle the case an empty set of conflicted_zipcodes
    elif use_only_combined_conflicted_zipcode:
        combined_conflicted_zipcodes = combined_conflicted_zipcodes
        conflicted_zipcodes = find_conflicting_zipcodes(data_json)
    else:
        conflicted_zipcodes = set(influence_constraints_zipcode.keys())

    print("Number of zipcodes for which constraints would be added: %d" % (len(conflicted_zipcodes)))

    overlap_vec = []
    z3_zipcode_constraint_vecs = {}
    for zipcode_idx, (zipcode, h) in enumerate(influence_constraints_zipcode.items()):
        if combined_conflicted_zipcodes is not None:
            if zipcode not in combined_conflicted_zipcodes:
                continue
            control_OR = Or([set_alloc[store_idx] == 0 for store_idx, store in h['control_influence']])
            test_OR = Or([set_alloc[store_idx] > 0 for store_idx, store in h['test_influence']])
            z3_zipcode_constraint_vecs[zipcode] = [control_OR, test_OR]
        else:
            if zipcode not in conflicted_zipcodes:
                continue
            control_OR = Or([set_alloc[store_idx] == 0 for store_idx, store in h['control_influence']])
            test_OR = Or([set_alloc[store_idx] > 0 for store_idx, store in h['test_influence']])
            z3_zipcode_constraint_vecs[zipcode] = [control_OR, test_OR]


        if zipcode not in conflicted_zipcodes:
            continue
        temp_overlap = Bool('zipcode_overlap_%s_%d' % (suffix_str, zipcode_idx))
        s.add(temp_overlap == (And(control_OR, test_OR)))
        overlap_vec.append(temp_overlap)

    num_overlaps = Real("num_overlaps"+suffix_str)
    if len(overlap_vec) > 0:
        s.add(num_overlaps >= 0)
        s.add(num_overlaps == Sum([If(i, 1, 0) for i in overlap_vec]))
    else:
        # this can happen if use_only_conflicted_zipcodes=True and there are no conflict zipcodes
        # we fix num_overlaps at 0, so the solver now just solves SAT
        print("Nothing to optimize since potential zip overlaps are 0. z3 will fallback to SAT.")
        s.add(num_overlaps == 0)



    # print(z3_zipcode_constraint_vecs)
    #z3_zipcode_constraint_vecs.append(s)
    #df = pd.DataFrame(z3_zipcode_constraint_vecs, columns=["colummn"])
    #df.to_csv('D:/project/cac/z3_explorations/assets/test.csv', index=False)
    #print("zipcode_constraint has been added")


    # return the solver object, without solving SAT, if requested
    if only_return_solver:
        return s, num_overlaps, z3_zipcode_constraint_vecs, set_alloc, ordered_stores, set_id_to_name

    print("Number of assertions created: %d" %(len(s.assertions())))
    # print(s.simplify())
    start_time = datetime.datetime.now()
    print("Beginning solution at %s." % (str(start_time)))

    if solve_as_SAT:
        s.add(num_overlaps <= SAT_upper_bound)
    else:
        s.minimize(num_overlaps)

    if s.check() == sat:
        finish_time = datetime.datetime.now()
        duration_sec = (finish_time - start_time).total_seconds()
        print("Found solution! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
        m = s.model()
        # print(m)
        subsets = defaultdict(list)
        subsets_with_name = defaultdict(list)
        for s_idx, store in enumerate(ordered_stores):
            subset_id = m[set_alloc[s_idx]].as_long()
            subsets[subset_id].append(store)
            subsets_with_name[set_id_to_name[subset_id]].append(store)
        split_utils.write_results_to_df(data_json, subsets_with_name, results_file,
                                        other_info="Runtime: %f sec" % (duration_sec,),
                                        use_size_attribute=use_size_info)
        # for k, v in sorted(subsets.items()):
        #     subset_size = len(v)
        #     subset_type = 'control' if k==0 else 'test'
        #     print("subset_%s, size=%d, %s: %s" % (k, subset_size, subset_type, ",".join(v)))
        return subsets_with_name
    elif s.check() == unsat:
        print("unsatisfiable condition")
    elif s.check() == unknown:
        print("unknown end condition")
    else:
        finish_time = datetime.datetime.now()
        duration_sec = (finish_time - start_time).total_seconds()
        print("No solution found. (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
