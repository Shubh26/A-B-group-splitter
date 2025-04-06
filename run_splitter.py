from datetime import datetime
#from pathlib import Path
import random, string, os, json
import atexit
import traceback
import sys
from pathlib import Path
import single_retailer_splitter_new as srs
import multi_retailer_splitter_new as mrs
import emailer_new as em

success = False
exception_msg = ""
DATE_FORMAT_TIMESTAMP='%Y%m%d%H%M%S'

def get_timestamp(date_format=DATE_FORMAT_TIMESTAMP):
    return datetime.utcnow().strftime(date_format)

def generate_random_unique_id(length=5):
    """
    Returns a unique random string
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def get_retailer(file):
    #retailer = os.path.abspath(file).split('\\')[-1].split('_')[0]
    retailer = Path(file).name.split('_')[0]
    return retailer


def generate_z3_folder(config_dict):
    z3_dir = 'z3_output'
    mode = config_dict['mode']
    cluster = config_dict['cluster']
    uid = generate_random_unique_id()
    for file in config_dict['z3_input_file']:
        if mode == 'multi':
            avg_tol = "_".join([str(_) for _ in config_dict['avg_tol']])
            size_tol = "_".join([str(_) for _ in config_dict['size_tol']])
            if cluster:
                groups = "_".join([str(_) for _ in config_dict['groups']])
                output_folder = os.path.join(os.path.dirname(file), z3_dir, f"{config_dict['mode']}_retailer_clustered", f"groups_{groups}_avgtol_{avg_tol}_sizetol_{size_tol}_{uid}")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            else:
                output_folder = os.path.join(os.path.dirname(file), z3_dir, f"{config_dict['mode']}_retailer_non_clustered", f"avgtol_{avg_tol}_sizetol_{size_tol}_{uid}")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

        if mode == 'single':
            if cluster:
                output_folder = os.path.join(os.path.dirname(file), z3_dir, f"{config_dict['mode']}_retailer_clustered", f"{get_retailer(file)}_groups_{config_dict['groups']}_avgtol_{config_dict['avg_tol']}_sizetol_{config_dict['size_tol']}_{generate_random_unique_id()}")
                print(output_folder)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            else:
                output_folder = os.path.join(os.path.dirname(file), z3_dir, f"{config_dict['mode']}_retailer_non_clustered", f"{get_retailer(file)}_avgtol_{config_dict['avg_tol']}_sizetol_{config_dict['size_tol']}_{generate_random_unique_id()}")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

    return output_folder


def run_splitter(campaign_details_json, z3_input_file, mode, cluster, split, groups, avg_tol, size_tol):
    with open(campaign_details_json) as f:
        campaign_details = json.loads(f.read())
    config_dict = {'z3_input_file': z3_input_file, 'mode': mode, 'cluster': cluster, 'split': split, 'groups': groups, 'avg_tol': avg_tol, 'size_tol': size_tol}
    print(config_dict)
    config_dict['output_folder'] = os.path.realpath(generate_z3_folder(config_dict))
    campaign_details.update(config_dict)
    #atexit.register(em.email_notifier, campaign_details, state="terminated")
    try:
        if campaign_details["mode"] == "single":
            srs.single_retailer_splitter(campaign_details)
        if campaign_details["mode"] == "multi":
            mrs.multi_retailer_splitter(campaign_details)
    except Exception as e:
        exception_msg = traceback.format_exc()
        raise e
    success = True

if __name__ == "__main__":
    campaign_details_json = sys.argv[1]
    z3_input_file = sys.argv[2]
    z3_input_file = json.loads(z3_input_file.replace("'", '"'))
    mode = sys.argv[3]
    cluster = eval(sys.argv[4])
    split = sys.argv[5]
    split = json.loads(split.replace("'", '"'))

    groups = sys.argv[6]
    if groups == "None" or groups == "[None]":
        groups = None
    elif type(eval(groups)) == list:
        groups = eval(groups)
    else:
        groups = int(groups)

    avg_tol = sys.argv[7]
    if type(eval(avg_tol)) == list:
        avg_tol = eval(avg_tol)
    else:    
        avg_tol = int(avg_tol)

    size_tol = sys.argv[8]
    if type(eval(size_tol)) == list:
        size_tol = eval(size_tol)
    else:
        size_tol = int(size_tol)

    #print(campaign_details_json, z3_input_file, mode, cluster, split, groups, avg_tol, size_tol)
    run_splitter(campaign_details_json, z3_input_file, mode, cluster, split, groups, avg_tol, size_tol)
