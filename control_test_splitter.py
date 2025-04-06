from z3 import *
set_option(verbose=10)
set_param('parallel.enable', True)

from collections import defaultdict
import json, datetime, itertools, os
from pathlib import Path
import pandas as pd, numpy as np

INPUT_DATA_FILE = r'sample_input_data.txt'


def z3_abs(x):
    return If(x >= 0, x, -x)


def count_val(vec, vec_size, val):
    return Sum([If(vec[i] == val, 1, 0) for i in range(vec_size)])


def elem_sum(alloc_vec, cost_vec, vec_size, alloc_id):
    return Sum([cost_vec[i] * If(alloc_vec[i]==alloc_id, 1, 0) for i in range(vec_size)])


def write_results_to_df(original_data, subsets, file_path, control_key='control', other_info=None,
                        use_size_attribute=False):
    """
    This is to make it easy to evaluate the output of the optimizer.
    The function writes out some aggregate stats in the csv format.
    :param original_data:
    :param subsets: this is a dict with key=subset name, value=list of stores.
    :param file_path:
    :param control_key: specify which key in the subsets dict is control. This allows to report overlap.
    :param other_info: misc. info that needs to be written out
    :return:
    """
    file_path = Path(file_path)
    ## creating parent directories
    file_path.parent.mkdir(parents=True,exist_ok=True)

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as xlsxwriter_obj:
        #xlsxwriter_obj = pd.ExcelWriter(file_path, engine='openpyxl')
        df = pd.DataFrame(columns=['subset_name', 'subset_size', 'subset_size_pct', 'subset_score_sum',
                                   'subset_score_avg', "stores"])

        if use_size_attribute:
            total_set_size = sum([sum([original_data[k]['size'] for k in v]) for v in subsets.values()])
        else:
            total_set_size = sum([len(v) for v in subsets.values()])
        for subset_name, assigned_elements in sorted(subsets.items()):
            if use_size_attribute:
                subset_size =sum([original_data[elem]['size'] for elem in assigned_elements])
            else:
                subset_size = len(assigned_elements)
            scores = [original_data[elem]['score'] for elem in assigned_elements]
            scores_sum = sum(scores)
            scores_avg = 1.0 * scores_sum /subset_size
            df = df.append({'subset_name': subset_name, 'subset_size': subset_size, 'subset_score_sum': scores_sum,
                            'subset_score_avg': scores_avg, 'subset_size_pct': 1.0 * subset_size / total_set_size,
                            'stores': "\n".join(map(str, assigned_elements))},
                           ignore_index=True)
        # df.to_csv(file_path, index_label='S.No.')

        df.to_excel(xlsxwriter_obj, sheet_name='allocations', index_label='S.No.')
        if control_key is not None and control_key in subsets:
            df = pd.DataFrame(columns=["stat_name", "stat_value"])
            print("Computing overlap with control=%s, this might take a while." % (control_key,))
            test_zips, control_zips = get_test_control_zips(original_data, subsets, control_key)
            overlapping_zips = control_zips.intersection(test_zips)
            if len(overlapping_zips) > 0:
                print("%d overlapping zips found, these zips overlap: %s" % (len(overlapping_zips),
                                                                         ",".join(map(str, overlapping_zips))))
            else:
                print("0 overlapping zips.")
            df = df.append({'stat_name': 'num_overlapping_zips', 'stat_value': len(overlapping_zips)}, ignore_index=True)
            df = df.append({'stat_name': 'overlapping_zips', 'stat_value': ",".join(map(str, overlapping_zips))},
                           ignore_index=True)
            if other_info:
                df = df.append({'stat_name': 'other_info', 'stat_value': str(other_info)},
                               ignore_index=True)
            df.to_excel(xlsxwriter_obj, sheet_name='stats', index=False)
            xlsxwriter_obj.save()
            print(f"*********done writing to excel {Path(file_path).absolute()}")




def write_multi_retailer_results_to_df(original_data, subsets, control_key='control', other_info=None,
                        use_size_attribute=False, retailer_idx = None):
    # (original_data, subsets, control_key='control', file_path, other_info=None,
    # use_size_attribute=False)
    """
    This is to make it easy to evaluate the output of the optimizer.
    The function writes out some aggregate stats in the csv format.
    :param original_data:
    :param subsets: this is a dict with key=subset name, value=list of stores.
    :param file_path:
    :param control_key: specify which key in the subsets dict is control. This allows to report overlap.
    :param other_info: misc. info that needs to be written out
    :return:
    """
    #file_path = Path(file_path)
    # creating parent directories
    #file_path.parent.mkdir(parents=True,exist_ok=True)

    # if xlsxwriter_obj is None:
    #     xlsxwriter_obj = pd.ExcelWriter(file_path, engine='xlsxwriter')
    #
    #xlsxwriter_obj = pd.ExcelWriter(file_path, engine='xlsxwriter')

    df1 = pd.DataFrame(columns=['retailer','subset_name', 'subset_size', 'subset_size_pct', 'subset_score_sum',
                               'subset_score_avg', "stores"])

    if use_size_attribute:
        total_set_size = sum([sum([original_data[k]['size'] for k in v]) for v in subsets.values()])
    else:
        total_set_size = sum([len(v) for v in subsets.values()])
    for subset_name, assigned_elements in sorted(subsets.items()):
        if use_size_attribute:
            subset_size =sum([original_data[elem]['size'] for elem in assigned_elements])
        else:
            subset_size = len(assigned_elements)
        scores = [original_data[elem]['score'] for elem in assigned_elements]
        scores_sum = sum(scores)
        scores_avg = 1.0 * scores_sum /subset_size
        df1 = df1.append({'retailer': f'retailer_{retailer_idx + 1}' ,'subset_name': subset_name, 'subset_size': subset_size, 'subset_score_sum': scores_sum,
                        'subset_score_avg': scores_avg, 'subset_size_pct': 1.0 * subset_size / total_set_size,
                        'stores': "\n".join(map(str, assigned_elements))},
                       ignore_index=True)
    # df.to_csv(file_path, index_label='S.No.')

    #df.to_excel(xlsxwriter_obj, sheet_name='allocations', index_label='S.No.')

    if control_key is not None and control_key in subsets:
        df2 = pd.DataFrame(columns=["retailer", "stat_name", "stat_value"])
        print("Computing overlap with control=%s, this might take a while." % (control_key,))
        test_zips, control_zips = get_test_control_zips(original_data, subsets, control_key)
        overlapping_zips = control_zips.intersection(test_zips)
        if len(overlapping_zips) > 0:
            print("%d overlapping zips found, these zips overlap: %s" % (len(overlapping_zips),
                                                                     ",".join(map(str, overlapping_zips))))
        else:
            print("0 overlapping zips.")
        df2 = df2.append({'retailer': f'retailer_{retailer_idx + 1}'}, ignore_index=True)
        df2 = df2.append({'stat_name': 'num_overlapping_zips', 'stat_value': len(overlapping_zips)}, ignore_index=True)
        df2 = df2.append({'stat_name': 'overlapping_zips', 'stat_value': ",".join(map(str, overlapping_zips))},
                       ignore_index=True)

        if other_info:
            df2 = df2.append({'stat_name': 'other_info', 'stat_value': str(other_info)},
                           ignore_index=True)
        #df.to_excel(xlsxwriter_obj, sheet_name='stats', index=False)
        #xlsxwriter_obj.save()
        #print(f"*********done writing to excel {Path(file_path).absolute()}")

        return df1, df2



def get_test_control_zips(original_data, subsets, control_key='control'):
    """
    Will return 2 groups - test zips, control zips

    Arguments:
    original_data:dict
        original input dictionary with store_id as key & value as another dictionary with keys 'control_influence', 'test_influence'
    subsets:dict
        a dict with key as name of the group & value as the store list for that group.
    control_key:string
        specify which key in the subsets dict is control.
    """
    control_zips, test_zips = set(), set()
    for subset_name, assigned_elements in subsets.items():
        temp_zips = []
        for elem in assigned_elements:
            temp_zips = original_data[elem]['control_influence' if subset_name == control_key else 'test_influence']
            if subset_name == control_key:
                control_zips.update(temp_zips)
            else:
                test_zips.update(temp_zips)
    return test_zips, control_zips


def multiple_retailers_sequential_test_control(list_of_data_jsons, list_of_splits, op_dir, avg_tol=5, size_tol=5,
                                              num_trials=3, wt_forbidden_zips=1.0, wt_seen_zips=1.0,
                                               generate_debug_info=False):
    """
    A different way to optimize for multiple retailers - we solve for one retailer first, and then use the generated
    test zipcodes as forbidden control zipcodes for the next retailer. This is obivously not very optimal. But it is
    fast since there are no cross constraints. This whole process is repeated num_trials times and the best response
    is returned.

    It is probably faster the arrange retailers in increasing order of # stores; earlier retailer generate
    lesser constraints so later retailers in the list, would have more freedom to be optimal.
    :param list_of_data_jsons:
    :param list_of_splits:
    :param op_dir:
    :param avg_tol:
    :param size_tol:
    :param generate_debug_info:
    :param wt_forbidden_zips: the multiplier for the penalty arising out of forbidden zipcodes; this is provided so that
        the relative weight of this penalty might be lowered, so as to not overwhelm the total penalty. Otherwise, we
        won't be able to optimize the per-retailer overlap a lot.
    :param wt_seen_zips:
    :param num_trials:
    :return:
    """

    num_retailers = len(list_of_data_jsons)
    print("Solve for %d retailers." % (num_retailers,))

    # check if retailers are ordered by size
    if not np.all(np.argsort([len(i) for i in list_of_data_jsons]) == range(num_retailers)):
        print("WARNING: it is faster to sort retailers in order of number of stores")

    df_results = pd.DataFrame(columns=['trial_num', 'cross_overlaps', 'sat_status'] +
                                      ['retail_%d_overlaps'%(i+1,) for i in range(num_retailers)])
    summary_results_file = op_dir + os.sep + 'results_summary.csv'
    # some standard symbols
    test_indicator = 1
    control_indicator = -1
    overlap_indicator = 1
    clean_indicator = 0

    # we need to maintain a set of control zips to be avoided for the *first* retailer, based on the solution produced
    # in the previous iterations. If this is not done, the first retailer would always produce the same allocations (z3
    # is not stochastic) in every trial run, rendering this exercise useless. We need to constrain the first retailer
    # only, since its store/zip allocations cascades to produce different solutions for the other retailers.
    seen_controls_zips_first_retailer = set()
    for trial_idx in range(num_trials):
        print("Trial number: %d" % (trial_idx + 1))
        sat_count = 0
        # create a directory for this iteration
        trial_dir = os.path.join(op_dir, "trial_%d" % (trial_idx+1))
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)
        # within a trial iteration we want to ensure the test zips from retailer i do not end up in the control for
        # retailer i+1
        forbidden_control_zipcodes = set()
        all_assigned_zipcodes = []
        for retailer_idx, (retailer_data, splits) in enumerate(zip(list_of_data_jsons, list_of_splits)):
            results_file = os.path.join(*[trial_dir, "results_retailer_%d.xlsx" % (retailer_idx + 1)])
            start_time = datetime.datetime.now()
            # NOTE: every inner loop has its own solver since we have no cross constraints
            current_solver = Optimize()

            # We don't really need the suffix string since we are instantiating a solver in the loop, but still keeping
            # it around since its safe for future/accidental changes. Also helpful for debugging.
            _, num_overlaps, zipcode_constraints, set_alloc, ordered_stores, set_id_to_name = set_splitter(
                data_json=retailer_data, splits=splits,
                results_file=results_file,
                avg_tol=avg_tol, size_tol=size_tol,
                use_soft_constraints=False,
                solver_obj=current_solver, only_return_solver=True,
                suffix_str="_%d_%d" % (trial_idx+1, retailer_idx + 1))

            sum_seen_overlaps, sum_forbidden_overlaps = None, None
            # modify solver to add constraints for forbidden zips
            common_zips = forbidden_control_zipcodes.intersection(set(zipcode_constraints.keys()))

            if len(common_zips) > 0:
                forbidden_overlaps = IntVector('forbidden_overlaps', len(common_zips))
                for zip_idx, common_zipcode in enumerate(common_zips):
                    zipcode_constr, num_constr = zipcode_constraints[common_zipcode]
                    current_solver.add(forbidden_overlaps[zip_idx]==If(count_val(zipcode_constr, num_constr,
                                                                                 control_indicator) > 0,
                                                                       overlap_indicator,
                                                                       clean_indicator))

                sum_forbidden_overlaps = Int('sum_forbidden_overlaps')
                current_solver.add(sum_forbidden_overlaps==Sum(forbidden_overlaps))

            if retailer_idx == 0 and len(seen_controls_zips_first_retailer) > 0:
                seen_overlaps = IntVector('seen_overlaps', len(seen_controls_zips_first_retailer))
                for zip_idx, seen_zipcode in enumerate(seen_controls_zips_first_retailer):
                    zipcode_constr, num_constr = zipcode_constraints[seen_zipcode]
                    current_solver.add(seen_overlaps[zip_idx] == If(count_val(zipcode_constr, num_constr,
                                                                                   control_indicator) > 0,
                                                                         overlap_indicator,
                                                                         clean_indicator))

                sum_seen_overlaps = Int('sum_seen_overlaps')
                current_solver.add(sum_forbidden_overlaps == Sum(seen_overlaps))

            if sum_forbidden_overlaps is not None:
                if sum_seen_overlaps is not None:
                    current_solver.minimize(num_overlaps +
                                            wt_forbidden_zips * ToReal(sum_forbidden_overlaps) +
                                            wt_seen_zips * ToReal(sum_seen_overlaps))
                else:
                    current_solver.minimize(num_overlaps + wt_forbidden_zips * ToReal(sum_forbidden_overlaps))
            else:
                current_solver.minimize(num_overlaps)

            print("Executing solver for retailer: %d" % (retailer_idx + 1))
            if current_solver.check() == sat:
                sat_count += 1
                finish_time = datetime.datetime.now()
                duration_sec = (finish_time - start_time).total_seconds()
                print("Found solution! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
                m = current_solver.model()
                subsets_with_name = defaultdict(list)
                for s_idx, store in enumerate(ordered_stores):
                    subset_id = m[set_alloc[s_idx]].as_long()
                    subsets_with_name[set_id_to_name[subset_id]].append(store)
                test_zips, control_zips = get_test_control_zips(retailer_data, subsets_with_name,
                                                                    control_key='control')
                forbidden_control_zipcodes.update(test_zips)
                if retailer_idx == 0:
                    seen_controls_zips_first_retailer.update(control_zips)
                all_assigned_zipcodes.append({'test': test_zips, 'control': control_zips})
                write_results_to_df(retailer_data, subsets_with_name, results_file, control_key='control')

            else:
                finish_time = datetime.datetime.now()
                duration_sec = (finish_time - start_time).total_seconds()
                print("SAT failed (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
                # breaking probably makes sense since one retailer unsat implies we actually have no solution in this
                # trial; so we move on to the next trial.
                break

        # summarize data for this trial
        # ['trial_num', 'cross_overlaps', 'sat_status', 'duration']
        all_tests, all_controls = set(), set()
        if len(all_assigned_zipcodes) == num_retailers:
            all_tests = set.union(*[info['test'] for info in all_assigned_zipcodes])
            all_controls = set.union(*[info['control'] for info in all_assigned_zipcodes])
            temp_data = {'trial_num': trial_idx + 1, 'sat_status': 'sat',
                        'cross_overlaps': len(all_tests.intersection(all_controls))}
            for info_idx, info in enumerate(all_assigned_zipcodes):
                retail_specific_overlap = len(set.intersection(set(info['test']), set(info['control'])))
                temp_data.update({'retail_%d_overlaps' % (info_idx + 1): retail_specific_overlap})
            df_results = df_results.append(temp_data, ignore_index=True)
        else:
            df_results = df_results.append({'trial_num': trial_idx + 1, 'sat_status': 'unsat for some retailer'},
                                           ignore_index=True)

    df_results.to_csv(summary_results_file, index=False)


def multiple_retailers_test_control(list_of_data_jsons, list_of_splits, op_dir, avg_tol=5, size_tol=5,
                                    use_asymmetric_constraints=False,
                                    generate_debug_info=False, solve_as_SAT=False, SAT_upper_bound=None):
    """
    *Experimental* code to deal with multiple retailers: not only constraints from individual retailers must be
    satisfied, but test zips of one retailer cannot be in the control zipcode set of another.
    :param list_of_data_jsons:
    :param list_of_splits:
    :param op_dir: dir to write out files for individual retailers
    :param avg_tol:
    :param size_tol:
    :param use_asymmetric_constraints: if True, for given retailers A and B, it avoids generating two sets of
                    constraints for mutual test control exclusion. It just generates one set, which leads to less number
                    of constraints, possibly leading to faster runtime.
    :param generate_debug_info: if True, writes out assertions in file op_dir/assertions.txt. This might generate a LOT
                    of data, so switch it on only when sure.
    :param solve_as_SAT: if you do not want to solve this as an optimize problem but a SAT problem. This might be
                helpful if some margin of error is acceptable on the zip overlaps, and you want to exit when you are
                below this.
    :param SAT_upper_bound: this variable is only considered if solve_as_SAT==True. This is the upper bound on the
                overlaps that decides the problem has found a SAT solution.
    :return:
    """
    if solve_as_SAT:
        if SAT_upper_bound is None:
            print("SAT upper bound cannot be None. Aborting.")
            return

    num_retailers = len(list_of_data_jsons)
    print("Solve for %d retailers." % (num_retailers,))

    # TODO: maybe we can have a "name mapper" that handles store name collisions implicitly and changes them back when
    # returning results. For now, this ensures we dont end up creating accidental constraints with the same names,
    # across retailers.
    # same_store_names = set.intersection(*[set(i.keys()) for i in list_of_data_jsons])
    # if len(same_store_names) > 0:
        # print("Multiple retailers data cannot have overlapping store names, %d overlaps found, aborting!" %
              # (len(same_store_names)))
        # return

    # create an universal solver and process each retailer's data one at a time
    if solve_as_SAT:
        universal_solver = Solver()
    else:
        universal_solver = Optimize()

    retail_specfic_zipcode_constraints = []
    retail_specfic_num_overlaps = []
    retail_specific_set_alloc = []
    retail_specific_ordered_stores = []
    retail_specific_set_id_to_name = []
    for retailer_idx, (retailer_data, splits) in enumerate(zip(list_of_data_jsons, list_of_splits)):
        # TODO: there should be one excel created in addition to the ones for individual retailers, to summarize
        # allocations across retailers.
        results_file = os.path.join(*[op_dir, "results_retailer_%d.xlsx"%(retailer_idx + 1)])
        print("Processing retailer %d of %d now." % (retailer_idx + 1, num_retailers))

        # the first returned value is the solver, which we dont need unless we are chaining
        _, num_overlaps, zipcode_constraints, set_alloc, ordered_stores, set_id_to_name = set_splitter(
                                                            data_json=retailer_data, splits=splits,
                                                            results_file=results_file,
                                                            avg_tol=avg_tol, size_tol=size_tol,
                                                            use_soft_constraints=False,
                                                            solver_obj=universal_solver, only_return_solver=True,
                                                            suffix_str=str(retailer_idx + 1))
        retail_specfic_num_overlaps.append(num_overlaps)
        retail_specfic_zipcode_constraints.append(zipcode_constraints)
        retail_specific_set_alloc.append(set_alloc)
        retail_specific_ordered_stores.append(ordered_stores)
        retail_specific_set_id_to_name.append(set_id_to_name)
        print("\tObtained modified solver.")
        print("\tNumber of assertions: %d" % (len(universal_solver.assertions())))

    # create the cross-retailer constraints here
    test_indicator = 1
    control_indicator = -1
    overlap_indicator = 1
    clean_indicator = 0
    cross_contraints_idx_combinations = []

    seen_combinations = []
    for test_retail_idx, control_retail_idx in itertools.product(range(num_retailers), range(num_retailers)):
        if test_retail_idx == control_retail_idx:
            continue
        if use_asymmetric_constraints:
            current_combination = {test_retail_idx, control_retail_idx}
            if current_combination in seen_combinations:
                continue
            else:
                seen_combinations.append(current_combination)
        cross_contraints_idx_combinations.append((test_retail_idx, control_retail_idx))

    num_cross_constraints = len(cross_contraints_idx_combinations)
    cross_overlap_vec = IntVector('cross_overlaps', num_cross_constraints)


    for combination_idx, (test_retail_idx, control_retail_idx) in enumerate(cross_contraints_idx_combinations):

        print("Adding cross constraint: test of retailer %d vs control of retailer %d" % (test_retail_idx+1,
                                                                                          control_retail_idx+1))
        test_retail_constr = retail_specfic_zipcode_constraints[test_retail_idx]
        control_retail_constr = retail_specfic_zipcode_constraints[control_retail_idx]
        common_zips = set.intersection(set(test_retail_constr.keys()), set(control_retail_constr.keys()))
        if len(common_zips) == 0:
            print(f"zero common zips between test of {test_retail_idx} & control of {control_retail_idx}")
            universal_solver.add(cross_overlap_vec[combination_idx] == clean_indicator)
        else:
            print(f"{len(common_zips)} common zips between test of {test_retail_idx} & control of {control_retail_idx}")
            common_zip_overlaps = IntVector('common_zips_overlap_vec_%d'%(combination_idx,), len(common_zips))

            for common_zip_idx, common_zip in enumerate(common_zips):

                common_zip_test_constr, num_test_constr = test_retail_constr[common_zip]
                common_zip_control_constr, num_control_constr = control_retail_constr[common_zip]

                universal_solver.add(common_zip_overlaps[common_zip_idx]==If(
                    And(count_val(common_zip_test_constr, num_test_constr, test_indicator) > 0,
                        count_val(common_zip_control_constr, num_control_constr, control_indicator) > 0),
                    overlap_indicator,
                    clean_indicator
                ))
            # add an aggregate score for the violation across common zipcodes for this test-control combination
            universal_solver.add(cross_overlap_vec[combination_idx] == Sum(common_zip_overlaps))


    # create a global minimizer
    all_overlaps = IntVector("all_overlaps", num_cross_constraints + num_retailers)
    for i in range(num_cross_constraints):
        universal_solver.add(all_overlaps[i] == cross_overlap_vec[i])
    for i in range(num_retailers):
        universal_solver.add(all_overlaps[num_cross_constraints + i] == retail_specfic_num_overlaps[i])

    start_time = datetime.datetime.now()
    print("Beginning solution at %s." % (str(start_time)))

    if solve_as_SAT:
        print("Objective is to SAT with bound=%d." % (SAT_upper_bound,))
        universal_solver.add(Sum(all_overlaps) <= SAT_upper_bound)
    else:
        print("Objective is to minimize.")
        universal_solver.minimize(Sum(all_overlaps))

    if generate_debug_info:
        with open(op_dir + os.sep + 'assertions.txt', 'w') as f_assert:
            for assertion in universal_solver.assertions():
                f_assert.write(str(assertion) + "\n")

    if universal_solver.check() == sat:
        finish_time = datetime.datetime.now()
        duration_sec = (finish_time - start_time).total_seconds()
        print("Found solution! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
        m = universal_solver.model()
        # print(m)
        # retail_specific_subsets = defaultdict(lambda: defaultdict(list))
        # retail_specific_subsets_with_name = defaultdict(lambda: defaultdict(list))
        retail_specific_subsets = []
        retail_specific_subsets_with_name = []
        for retailer_idx, (ordered_stores, set_alloc, subsets_with_name, data_json) in enumerate(zip(retail_specific_ordered_stores, retail_specific_set_alloc, retail_specific_set_id_to_name, list_of_data_jsons)):
            subsets = defaultdict(list)
            subsets_with_name = defaultdict(list)
            for s_idx, store in enumerate(ordered_stores):
                subset_id = m[set_alloc[s_idx]].as_long()
                subsets[subset_id].append(store)
                subsets_with_name[set_id_to_name[subset_id]].append(store)
            retail_specific_subsets.append(subsets)
            retail_specific_subsets_with_name.append(subsets_with_name)
            results_file = os.path.join(op_dir,f'retailer_{retailer_idx}.xlsx')
            write_results_to_df(data_json, subsets_with_name, results_file, other_info="Runtime: %f sec" % (duration_sec,))

        # for k, v in sorted(subsets.items()):
        #     subset_size = len(v)
        #     subset_type = 'control' if k==0 else 'test'
        #     print("subset_%s, size=%d, %s: %s" % (k, subset_size, subset_type, ",".join(v)))
        return retail_specific_subsets_with_name
    elif universal_solver.check() == unsat:
        print("unsatisfiable condition")
        # print(f'unstat core ***\n{universal_solver.unsat_core()}')
        # print(f'unstat core ***\n{universal_solver.unsat_core}')
    elif universal_solver.check() == unknown:
        print("unknown end condition")
    else:
        finish_time = datetime.datetime.now()
        duration_sec = (finish_time - start_time).total_seconds()
        print("No solution found. (at=%s, total=%f sec)" % (str(finish_time), duration_sec))


def set_splitter(data_json, splits, results_file, avg_tol=5, size_tol=5, use_soft_constraints=False,
                 solver_obj=None, only_return_solver=False, suffix_str=None):
    """
    Assumption: one control group, possibly more than one test group
    :param data_json:
    :param splits: a dict with entries for split fractions of subsets. It is mandatory to have one key as "control"
    :param results_file: this is where results would be written as csv
    :param avg_tol: how much can the subset scores averages differ by
    :param size_tol: how much can subset sizes differ by, beyond what is expected from the splits
    :param solver_obj: pass in an existing solver object to which constraints are to be added.
    :param only_return_solver: don't solve, and importantly do not add the minimization constraint - only return solver
    :param suffix_str: if not None or greater than zero length, is appended to zipcode var names inside z3 - this is
        to avoid naming conflicts in case caller needs to separate out same zipcodes across calls in the same solver obj
    :return:
    """
    assert 'control' in splits
    assert sum(splits.values()) == 1
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
    set_alloc = IntVector('set_alloc'+suffix_str, num_elems)

    # reuse a solver object if one is passed in
    s = Optimize() if solver_obj is None else solver_obj
    for i in range(num_elems):
        s.add(And(set_alloc[i] > -1, set_alloc[i] < num_sets))

    # Create some common collections - for setting constraints later
    # This probably could be replaced with a native type, but to be safe, I am letting this stay,
    # till we have a test setup.
    expected_subset_sizes = RealVector('expected_subset_sizes'+suffix_str, num_sets)
    current_subset_sizes = IntVector('current_subset_sizes'+suffix_str, num_sets)
    current_subset_scores = RealVector('current_subset_scores'+suffix_str, num_sets)
    for i in range(num_sets):
        s.add(expected_subset_sizes[i] == splits[set_id_to_name[i]] * num_elems)
        s.add(current_subset_sizes[i] == count_val(set_alloc, num_elems, i))
        s.add(current_subset_scores[i] == elem_sum(set_alloc, ordered_costs, num_elems, i))

    # size constraints - compare expected size to actual size
    for i in range(num_sets):
        if use_soft_constraints:
            s.add_soft(z3_abs(current_subset_sizes[i] - expected_subset_sizes[i]) <= size_tol)
        else:
            s.add(z3_abs(current_subset_sizes[i] - expected_subset_sizes[i]) <= size_tol)

    # subset average score constraints - averages must much to a tolerance
    control_avg_score = Real('control_avg_score'+suffix_str)
    s.add(control_avg_score == current_subset_scores[0]/current_subset_sizes[0])
    for i in range(1, num_sets):
        if use_soft_constraints:
            s.add_soft(z3_abs(current_subset_scores[i]/current_subset_sizes[i] - control_avg_score) <= avg_tol)
        else:
            s.add(z3_abs(current_subset_scores[i] / current_subset_sizes[i] - control_avg_score) <= avg_tol)

    # create zipcode constraints - one per influence area for a store
    test_indicator = 1  # if zipcode belongs to ANY test group
    control_indicator = -1
    dummy_indicator = 0
    zipcode_constraints = defaultdict(list)
    for store_idx, store in enumerate(ordered_stores):
        for zipcode in data_json[store]['control_influence']:
            zipcode_constraints[zipcode].append(If(set_alloc[store_idx]==0, control_indicator, dummy_indicator))
        for zipcode in data_json[store]['test_influence']:
            zipcode_constraints[zipcode].append(If(set_alloc[store_idx] > 0, test_indicator, dummy_indicator))

    # now create an IntVector per zipcode using the above relationships
    overlap_indicator = 1  # zipcode is part of control and some test group
    clean_indicator = 0  # zipcode is part of control or some test group but not both

    overlap_vec = IntVector('overlaps'+suffix_str, len(zipcode_constraints))
    z3_zipcode_constraint_vecs = {}
    for zipcode_idx, (zipcode, constraints) in enumerate(zipcode_constraints.items()):
        zipcode_constr_vec = IntVector("%s%s" % (zipcode, suffix_str), len(constraints))
        for c_idx, c in enumerate(constraints):
            s.add(zipcode_constr_vec[c_idx]==c)
        s.add(overlap_vec[zipcode_idx] == If(
                                            And(count_val(zipcode_constr_vec, len(constraints), test_indicator) > 0,
                                                count_val(zipcode_constr_vec, len(constraints), control_indicator) > 0),
                                            overlap_indicator,
                                            clean_indicator))
        z3_zipcode_constraint_vecs[zipcode] = (zipcode_constr_vec, len(constraints))

    num_overlaps = Real("num_overlaps"+suffix_str)
    s.add(num_overlaps==Sum(overlap_vec))

    # return the solver object, without solving SAT, if requested
    if only_return_solver:
        return s, num_overlaps, z3_zipcode_constraint_vecs, set_alloc, ordered_stores, set_id_to_name

    print("Number of assertions created: %d" %(len(s.assertions())))
    # print(s.simplify())
    start_time = datetime.datetime.now()
    print("Beginning solution at %s." % (str(start_time)))
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
        write_results_to_df(data_json, subsets_with_name, results_file, other_info="Runtime: %f sec" % (duration_sec,))
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


if __name__ == "__main__":
    # splits = {'control': 0.25, 'test': 0.75}
    # with open(INPUT_DATA_FILE) as f:
        # sample_data = json.loads(f.read())
    # with open(r'./scratch/jewel_store.json') as f:
        # jewel_data = json.loads(f.read())
    # set_splitter(sample_data, splits, size_tol=0, avg_tol=5000, results_file=r'./scratch/alloc',
    #              use_soft_constraints=True)
    # set_splitter(jewel_data, splits, size_tol=10, avg_tol=10, results_file=r'./scratch/jewel_allocations.xlsx')
    # list_of_data_jsons = [jewel_data]*2
    # list_of_splits = [splits]*2
    # op_dir=r'./scratch/multi_retailer_test'
    if len(sys.argv)!=2:
        print('Usage: python control_test_splitter.py <config_file>\n')
        sys.exit(1)
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.loads(f.read())
    file_paths =  config['file_paths']
    list_of_splits = config['splits']
    list_of_data_jsons = []
    for file_path in file_paths:
        with open(file_path) as f:
            data = json.loads(f.read())
            list_of_data_jsons.append(data)
    op_dir = config["output_folder"]
    avg_tol = config["avg_tol"]
    size_tol = config["size_tol"]
    if len(file_paths)>1:
        print("mulitple files passed running multi retailer code")
        retail_specific_subsets_with_name = multiple_retailers_test_control(list_of_data_jsons, list_of_splits, op_dir, avg_tol=avg_tol, size_tol=size_tol)
    elif len(file_paths) == 1:
        data = list_of_data_jsons[0]
        splits = list_of_splits[0]
        results_file = os.path.join(op_dir,'output_file.xlsx')
        print("only 1 file passed ,running set splitter")
        subsets = set_splitter(data, splits, size_tol=size_tol, avg_tol=avg_tol, results_file=results_file, use_soft_constraints=False)
    else:
        print("pass atleast one file as input")

