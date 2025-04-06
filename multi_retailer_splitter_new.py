from z3 import *
import json, os, itertools, glob, datetime
import pandas as pd
import optimized_splitter as ots
import control_test_splitter as split_utils
from collections import defaultdict
import emailer_new as em
import run_splitter as rs



def get_subset_cross_zips_overlap (list_of_data_jsons, list_of_subsets_zips):
    cross_overlapping_zip = []
    seen_combinations = []
    cross_overlapping_zip_dict = {}
    retailer = 0
    for retailer_x, retailer_y in itertools.product(range(len(list_of_data_jsons)), range(len(list_of_data_jsons))):
        if retailer_x == retailer_y:
            continue
        current_combination = {retailer_x, retailer_y}
        if current_combination in seen_combinations:
            continue
        else:
            seen_combinations.append(current_combination)

        print("checking cross overlap zips between retailer_%d and retailer_%d" % (retailer_x + 1, retailer_y + 1))
        test_x, control_x = list_of_subsets_zips[retailer_x]
        test_y, control_y = list_of_subsets_zips[retailer_y]
        cross_overlapping_zip.append(set.union(test_x.intersection(control_y), test_y.intersection(control_x)))
        retailer += 1
        print("%d zips overlap between retailer_%d and retailer_%d, zips: "
              % ((len(cross_overlapping_zip[retailer - 1]), retailer_x + 1, retailer_y + 1)), cross_overlapping_zip[retailer - 1])
        cross_overlapping_zip_dict[f'retailer_{retailer_x + 1} and retailer_{retailer_y + 1}'] = cross_overlapping_zip[retailer - 1]

    return cross_overlapping_zip_dict


def find_combined_conflicting_zipcodes(list_of_data_jsons):
    print("Beginning analysis for combined conflicting zipcodes.")
    all_zipcodes_influence = []
    for retailer_idx, retailer in enumerate(list_of_data_jsons):
        zipcodes_influence = defaultdict(lambda: defaultdict(set))
        subkeys = ['control_influence', 'test_influence']
        for store, info in retailer.items():
            for sk in subkeys:
                for z in info[sk]:
                    if z == 'null':
                        continue
                    zipcodes_influence[z][sk].add(store)
            print("Total no. of stores in retailer %d" %(len(retailer.keys())))
        print("Total zipcodes for retailer %d: %d" % (retailer_idx + 1, len(zipcodes_influence)))
        all_zipcodes_influence.append(zipcodes_influence)

    all_conflict_zipcodes = []
    for zipcodes_influence_idx, zipcodes_influence in enumerate(all_zipcodes_influence):
        conflict_zipcodes = []
        for z, h in zipcodes_influence.items():
            if len(h['control_influence']) == 0 or len(h['test_influence']) == 0:
                continue
            elif len(h['control_influence']) == 1 and h['control_influence'] == h['test_influence']:
                continue
            else:
                conflict_zipcodes.append(z)
        print("Conflicted zipcodes for retailer %d: %d (%0.02f%%)" %
              (zipcodes_influence_idx + 1, len(conflict_zipcodes),
               100.0 * len(conflict_zipcodes) / len(zipcodes_influence)))
        all_conflict_zipcodes.append(set(conflict_zipcodes))
    all_within_conflict_zipcodes = all_conflict_zipcodes.copy()

    seen_combinations = []
    all_cross_conflict_zipcodes = []
    retailer = -1
    for retailer_x, retailer_y in itertools.product(range(len(list_of_data_jsons)), range(len(list_of_data_jsons))):
        if retailer_x == retailer_y:
            continue
        current_combination = {retailer_x, retailer_y}
        if current_combination in seen_combinations:
            continue
        else:
            seen_combinations.append(current_combination)
        retailer += 1
        print("Checking common zipcodes between retailer_%d and retailer_%d" % (retailer_x + 1, retailer_y + 1))
        common_zips = set.intersection(set(all_zipcodes_influence[retailer_x].keys()),
                                       set(all_zipcodes_influence[retailer_y].keys()))
        print("Common zipcodes between retailer_%d and retailer_%d: %d (%.2f%% and %.2f%%)" % (
        retailer_x + 1, retailer_y + 1, len(common_zips),
        100.0 * len(common_zips) / len(all_zipcodes_influence[retailer_x].keys()),
        100.0 * len(common_zips) / len(all_zipcodes_influence[retailer_y].keys())))

        cross_conflict_zipcodes = []
        for z1, s1 in all_zipcodes_influence[retailer_x].items():
            for z2, s2 in all_zipcodes_influence[retailer_y].items():
                if z1 == z2:
                    if (len(s1['control_influence']) == 0 or len(s2['test_influence']) == 0) and (
                            len(s2['control_influence']) == 0 or len(s1['test_influence']) == 0):
                        continue
                    else:
                        cross_conflict_zipcodes.append(z1)
                else:
                    continue
        all_cross_conflict_zipcodes.append(cross_conflict_zipcodes)

        all_conflict_zipcodes[retailer_x] = set(cross_conflict_zipcodes).union(all_conflict_zipcodes[retailer_x])
        all_conflict_zipcodes[retailer_y] = set(cross_conflict_zipcodes).union(all_conflict_zipcodes[retailer_y])

        print("Revised conflicted zipcodes for retailer %d: %d (%0.02f%%)" % (
        retailer_x + 1, len(all_conflict_zipcodes[retailer_x]),
        100.0 * len(all_conflict_zipcodes[retailer_x]) / len(all_zipcodes_influence[retailer_x])))
        print("Revised conflicted zipcodes for retailer %d: %d (%0.02f%%)" % (
        retailer_y + 1, len(all_conflict_zipcodes[retailer_y]),
        100.0 * len(all_conflict_zipcodes[retailer_y]) / len(all_zipcodes_influence[retailer_y])))

        print("Cross conflicted zipcode between retailer %d and retailer %d: %d (%.2f%% and %.2f%%)" % (
        retailer_x + 1, retailer_y + 1, len(all_cross_conflict_zipcodes[retailer]),
        100.0 * len(all_cross_conflict_zipcodes[retailer]) / len(all_zipcodes_influence[retailer_x].keys()),
        100.0 * len(all_cross_conflict_zipcodes[retailer]) / len(all_zipcodes_influence[retailer_y].keys())))

    # return all_zipcodes_influence, all_conflict_zipcodes, all_cross_conflict_zipcodes
    #return
    return all_conflict_zipcodes, all_cross_conflict_zipcodes

def multi_retailer_splitter(config_dict):

    #h = open(r'/data/users/shubhamg/campaign/campaign_automation/testing/pepsico_dew_dor_pitchblack_jan23_v3/pepsico_dew_dor/testing_v2.log', 'a')
    list_of_data_jsons = []
    list_of_retailers = []
    for file in config_dict['z3_input_file']:
        list_of_retailers.append(rs.get_retailer(file))
        with open(file) as f:
            list_of_data_jsons.append(json.loads(f.read()))
    list_of_splits = config_dict['split']
    list_of_groups = config_dict['groups']
    list_of_avg_tol = config_dict['avg_tol']
    list_of_size_tol = config_dict['size_tol']
    # op_dir = config_dict['output_folder']

    solve_as_SAT = False
    SAT_upper_bound = None
    if solve_as_SAT:
        if SAT_upper_bound is None:
            print("SAT upper bound cannot be None. Aborting.")
            return

    if solve_as_SAT:
        universal_solver = Solver()
    else:
        universal_solver = Optimize()

    num_retailers = len(list_of_data_jsons)
    print("Solving for %d retailers." % (num_retailers))

    retail_specfic_zipcode_constraints = []
    retail_specfic_num_overlaps = []
    retail_specific_set_alloc = []
    retail_specific_ordered_stores = []
    retail_specific_set_id_to_name = []

    #retail_specific_set_splitter_output_file = []
    retail_specific_cluster_expansion_to_store = []
    retail_specific_final_sets_viz_file = []
    retail_specific_cluster_name_to_store_map = []
    retail_specific_store_centroids = []
    retail_specific_merged_data = []

    all_conflict_zipcodes, all_cross_conflict_zipcodes = find_combined_conflicting_zipcodes(list_of_data_jsons)

    results_file = os.path.join(config_dict['output_folder'], f"ab_split.xlsx")
    em.email_notifier(config_dict, "started")
    for retailer_idx, (retailer_data, splits) in enumerate(zip(list_of_data_jsons, list_of_splits)):
        print("Processing retailer %d of %d now: %s" % (retailer_idx + 1, num_retailers, list_of_retailers[retailer_idx]))

        combined_conflicted_zipcodes = all_conflict_zipcodes[retailer_idx]
        avg_tol = list_of_avg_tol[retailer_idx]
        size_tol = list_of_size_tol[retailer_idx]

        if config_dict['cluster']:
            groups = list_of_groups[retailer_idx]
            zipcode_to_latlong = ots.load_zipcodes()
            grp_txt = "_".join([str(_) for _ in config_dict['groups']])
            at_txt = "_".join([str(_) for _ in config_dict['avg_tol']])
            st_txt = "_".join([str(_) for _ in config_dict['size_tol']])

            cluster_viz_file = os.path.join(config_dict['output_folder'], f"{list_of_retailers[retailer_idx]}_clustered_graph.png")
            set_splitter_output_file = results_file
            cluster_expansion_to_store = os.path.join(config_dict['output_folder'], f"{list_of_retailers[retailer_idx]}_exptostores_groups_{grp_txt}_avg_tol_{at_txt}_size_tol_{st_txt}.json")
            final_sets_viz_file = os.path.join(config_dict['output_folder'], f"{list_of_retailers[retailer_idx]}_split_graph.png")

            retail_specific_cluster_expansion_to_store.append(cluster_expansion_to_store)
            retail_specific_final_sets_viz_file.append(final_sets_viz_file)

            n_groups, store_centroids = ots.group_stores(retailer_data, zipcode_to_latlong, n_groups=groups, plot_file=cluster_viz_file, distance_metric='euclidean')
            retail_specific_store_centroids.append(store_centroids)

            merged_data, cluster_name_to_store_map = ots.merge_stores(retailer_data, n_groups)
            retail_specific_merged_data.append(merged_data)
            retail_specific_cluster_name_to_store_map.append(cluster_name_to_store_map)

            s, num_overlaps, zipcode_constraints, set_alloc, ordered_stores, set_id_to_name = ots.set_splitter(data_json=merged_data, splits=splits,
                                                                                                           results_file=set_splitter_output_file, avg_tol=avg_tol, size_tol=size_tol,
                                                                                                           solver_obj=universal_solver, only_return_solver=True,
                                                                                                           suffix_str=str(retailer_idx + 1),
                                                                                                           use_size_info=True,
													   use_only_conflicted_zipcodes=False,
                                                                                                           use_only_combined_conflicted_zipcode=True,
                                                                                                           combined_conflicted_zipcodes=combined_conflicted_zipcodes)
        else:
            s, num_overlaps, zipcode_constraints, set_alloc, ordered_stores, set_id_to_name = ots.set_splitter(data_json=retailer_data, splits=splits,
                                                                                                           results_file=results_file,avg_tol=avg_tol, size_tol=size_tol,
                                                                                                           solver_obj=universal_solver, only_return_solver=True,
                                                                                                           suffix_str=str(retailer_idx + 1),
													   use_only_conflicted_zipcodes=False,
                                                                                                           use_only_combined_conflicted_zipcode=True,
                                                                                                           combined_conflicted_zipcodes=combined_conflicted_zipcodes)

        retail_specfic_num_overlaps.append(num_overlaps)
        retail_specfic_zipcode_constraints.append(zipcode_constraints)
        retail_specific_set_alloc.append(set_alloc)
        retail_specific_ordered_stores.append(ordered_stores)
        retail_specific_set_id_to_name.append(set_id_to_name)
        print("\tObtained modified solver.")
        print("\tNumber of assertions: %d" % (len(universal_solver.assertions())))
        print("\tadding retailers specific constraints")

    seen_combinations = []
    common_zips_overlap_vec = []

    num_common_zips_overlap = Real("num_common_zips_overlap")
    retailer=-1

    for retailer_x, retailer_y in itertools.product(range(len(list_of_data_jsons)), range(len(list_of_data_jsons))):
        if retailer_x == retailer_y:
            continue
        current_combination = {retailer_x, retailer_y}
        if current_combination in seen_combinations:
            continue
        else:
            seen_combinations.append(current_combination)
        print("checking zipcode constraints of retailer_%d and retailer_%d" % (retailer_x+1, retailer_y+1))
        retailer+=1

        common_zips = all_cross_conflict_zipcodes[retailer]

        if len(common_zips) > 0:
            print("adding cross constraints")
            for common_zip_idx, common_zip in enumerate(common_zips):
                control_x, test_x = retail_specfic_zipcode_constraints[retailer_x][common_zip]
                control_y, test_y = retail_specfic_zipcode_constraints[retailer_y][common_zip]

                common_zips_overlap = Bool(
                    'common_zips_overlap_vec_retailer_%d_&_%d_%d' % (retailer_x + 1, retailer_y + 1, common_zip_idx))

                universal_solver.add(common_zips_overlap == Or(And(control_x, test_y), And(control_y, test_x)))
                common_zips_overlap_vec.append(common_zips_overlap)

            print("cross constraints added for retailer_%d and retailer_%d" % (retailer_x + 1, retailer_y + 1))

        else:
            continue

    if len(common_zips_overlap_vec) > 0:
        universal_solver.add(num_common_zips_overlap >= 0)
        universal_solver.add(num_common_zips_overlap == Sum([If(i, 1, 0) for i in common_zips_overlap_vec]))
        print("num common zip overlap constraint added")
    else:
        print("No common zips overlap")
        universal_solver.add(num_common_zips_overlap == 0)

    print("Begin optimizing.....")
    start_time = datetime.datetime.now()
    print("Beginning solution at %s." % (str(start_time)))

    total = Real("total overlaps")
    universal_solver.add(total == num_common_zips_overlap + Sum([i for i in retail_specfic_num_overlaps]))

    if solve_as_SAT is False:
         universal_solver.minimize(total)
    else:
        universal_solver.add(total <= SAT_upper_bound)

    if universal_solver.check() == sat:
            finish_time = datetime.datetime.now()
            duration_sec = (finish_time - start_time).total_seconds()
            print("Found solution! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
            m = universal_solver.model()
            #print(m)
            #h.close()
            retail_specific_subsets = []
            retail_specific_subsets_with_name = []
            retail_specific_subsets_zips = []
            cross_overlap_zips = []
            df = pd.DataFrame()
            temp1 = pd.DataFrame()
            temp2 = pd.DataFrame()
            temp3 = {}
            df_allocation = pd.DataFrame()
            df_stats = pd.DataFrame()
            xlsxwriter_obj = pd.ExcelWriter(results_file, engine='xlsxwriter')

            use_size_info = False
            if config_dict['cluster']:
                list_of_data_jsons = retail_specific_merged_data
                use_size_info = True


            for retailer_idx, (ordered_stores, set_alloc, set_id_to_name, data_json) in enumerate(zip(retail_specific_ordered_stores,
                                                                                                         retail_specific_set_alloc, retail_specific_set_id_to_name,
                                                                                                         list_of_data_jsons)):
                subsets = defaultdict(list)
                subsets_with_name = defaultdict(list)
                for s_idx, store in enumerate(ordered_stores):
                    subset_id = m[set_alloc[s_idx]].as_long()
                    subsets[subset_id].append(store)
                    subsets_with_name[set_id_to_name[subset_id]].append(store)
                retail_specific_subsets.append(subsets)
                retail_specific_subsets_with_name.append(subsets_with_name)

                #if not cluster_mode:
                #print('data json', data_json)
                #print('subset', subsets_with_name)
                #print('user_size_info', use_size_info)
                temp1, temp2 = split_utils.write_multi_retailer_results_to_df(data_json, subsets_with_name, other_info="Runtime: %f sec" % (duration_sec,), use_size_attribute=use_size_info, retailer_idx=retailer_idx)
                df_allocation = df_allocation.append(temp1)
                df_stats = df_stats.append(temp2)
                retail_specific_subsets_zips.append(split_utils.get_test_control_zips(data_json, subsets_with_name, control_key = 'control'))

            #if not cluster_mode:
            df_allocation.to_excel(xlsxwriter_obj, sheet_name='allocations', index_label='S.No.')
            df_stats.to_excel(xlsxwriter_obj, sheet_name='stats', index=False)

            temp3 = get_subset_cross_zips_overlap(list_of_data_jsons, retail_specific_subsets_zips)
            df_cross_overlap_zips = pd.DataFrame(columns=["retailer", "stat_name", "stat_value"])
            for retailer, cross_overlap_zips in temp3.items():
                df_cross_overlap_zips = df_cross_overlap_zips.append({'retailer': retailer}, ignore_index=True)
                df_cross_overlap_zips = df_cross_overlap_zips.append({'stat_name': 'num_cross_overlapping_zips', 'stat_value': len(cross_overlap_zips)},
                                 ignore_index=True)
                df_cross_overlap_zips = df_cross_overlap_zips.append({'stat_name': 'cross_overlapping_zips', 'stat_value': ",".join(map(str, cross_overlap_zips))},
                                 ignore_index=True)

            df_cross_overlap_zips.to_excel(xlsxwriter_obj, sheet_name='cross overlap stats', index=False)
            xlsxwriter_obj.save()
            print("writing files")

            if config_dict['cluster']:
                for idx, subsets_with_name in enumerate(retail_specific_subsets_with_name):
                    store_splits = ots.expand_groups_into_stores(subsets_with_name, retail_specific_cluster_name_to_store_map[idx])
                    pd.DataFrame.from_dict(store_splits, orient='index').transpose().to_json(retail_specific_cluster_expansion_to_store[idx])
                    ots.plot_store_sets(store_splits, retail_specific_store_centroids[idx], plot_file=retail_specific_final_sets_viz_file[idx], control_key='control')
            em.email_notifier(config_dict, "finished")
            return retail_specific_subsets_with_name

    elif universal_solver.check() == unsat:
            print("unsatisfiable condition")
    elif universal_solver.check() == unknown:
            print("unknown end condition")
    else:
            finish_time = datetime.datetime.now()
            duration_sec = (finish_time - start_time).total_seconds()
            print("No solution found. (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
