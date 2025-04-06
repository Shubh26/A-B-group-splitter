import optimized_splitter as ots
import json, os
import seaborn as sns; sns.set()
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import re


import time
import smtplib
import imghdr
from email.message import EmailMessage
import glob
import atexit
import random, string
import traceback


ASSETS_DIR_RELPATH = 'assets'
SCRATCH_DIR_RELPATH = 'scratch'
DATE_FORMAT_TIMESTAMP='%Y%m%d%H%M%S'
success = False
exception_msg = ""
# create a scratch dir
scratch_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), SCRATCH_DIR_RELPATH)
if not os.path.exists(scratch_dir):
    print("Creating scratch dir at: %s" % (scratch_dir,))
    os.mkdir(scratch_dir)
    print("Created.")

def email_notification_exit(output_folder, file_name, n_groups, avg_tol, size_tol):
#def email_notification_exit(n_groups):
    #print("entered termination")
    campaign, store = get_campaign_info(file_name)
    t = f'z3 jobs has been terminated for {campaign} campaign for {store} with these parameters: {n_groups} groups, {avg_tol} avg_tol and {size_tol} size_tol at this location {output_folder}, exception message is {exception_msg}'
    if not success:
        EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
        EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

        contacts = ['shubham.gupta@247.ai', 'vikram.gupta@247.ai','sahana.ka@247.ai']
        #contacts = ['shubham.gupta@247.ai']

        msg = EmailMessage()
        msg['Subject'] = f'{os.getlogin()} z3 jobs for {campaign} campaign {store} store has been terminated'
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = contacts
        #t = f'z3 jobs has been terminated for {campaign} campaign for {store} with these parameters: {n_groups} groups, {avg_tol} avg_tol and {size_tol} size_tol at this location {output_folder}, exception message is {exception_msg}'
        with open(os.path.join(output_folder, 'failure_msg.txt'),'w') as f:
            f.write(t)
        #t = f'{n_groups} process terminated'
        msg.set_content(t)
        with smtplib.SMTP('mail.svc.cloud.247-inc.net', port = 25) as smtp:
                smtp.send_message(msg)
    else:
        with open(os.path.join(output_folder, 'success_msg.txt'),'w') as f:
                f.write(t)


def get_campaign_info(file_name):
    campaign = re.split("[_.]", file_name)[0]
    store = re.split("[_.]", file_name)[-2]
    return campaign, store


def email_notification(output_folder, campaign, store, n_groups, avg_tol, size_tol):
    EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

    contacts = ['shubham.gupta@247.ai', 'vikram.gupta@247.ai','sahana.ka@247.ai']
    #contacts = ['shubham.gupta@247.ai']

    msg = EmailMessage()
    msg['Subject'] = f'{campaign} z3 jobs for {store}'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = contacts
    t = f'z3 jobs has been started for {campaign} campaign for {store} with these parameters: {n_groups} groups, {avg_tol} avg_tol and {size_tol} size_tol'
    msg.set_content(t)
    with smtplib.SMTP('mail.svc.cloud.247-inc.net', port = 25) as smtp:
            smtp.send_message(msg)


def email_group_info(output_folder, campaign, store, n_groups, avg_tol, size_tol):
    EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

    contacts = ['shubham.gupta@247.ai', 'vikram.gupta@247.ai','sahana.ka@247.ai']
    #contacts = ['shubham.gupta@247.ai']


    msg = EmailMessage()
    msg['Subject'] = f'{campaign} z3 jobs for {store}'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = contacts
    t = f'here are the split details generated for {campaign} campaign for {store} from these parameters: {n_groups} groups, {avg_tol} avg_tol and {size_tol} size_tol'
    msg.set_content(t)


    files = glob.glob(os.path.join(output_folder, r'demo_splits_with*'))

    # files = glob.glob(r'/data/users/shubhamg/z3_explorations/scratch/400_20220125140720/demo_splits_with*',
    #                    recursive = True)


    for file in files:
        t = file.split('/')
        filename = t[-1]
        filetype = filename.split('.')
        filetype = filetype[-1]
        if filetype == 'png':
            with open (file, 'rb') as f:
                file_data = f.read()
                image_type = imghdr.what(f.name)
            msg.add_attachment(file_data, maintype = 'image', subtype=image_type, filename = filename)

        else:
            with open (file, 'rb') as f:
                file_data = f.read()
            msg.add_attachment(file_data, maintype = 'appplication', subtype='octet-stream', filename = filename)


    #context=ssl.create_default_context()

    with smtplib.SMTP('mail.svc.cloud.247-inc.net', port = 25) as smtp:
            smtp.send_message(msg)

#def run_splitter(cluster, file_name, splits, avg_tol, size_tol, n_groups=None, results_file=None):
#    if cluster == "Yes":
#       demo_grouping(file_name, n_groups, splits, avg_tol, size_tol)
#    else:
#       ots.set_splitter(file_name, splits, results_file, avg_tol, size_tol)


def get_timestamp(date_format=DATE_FORMAT_TIMESTAMP):
        return datetime.utcnow().strftime(date_format)

def generate_random_unique_id(length=5):
    """
    Returns a unique random string
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def create_output_folder(file_name, n_groups, avg_tol, size_tol):
    campaign = re.split("[_.]", file_name)[0]
    store = re.split("[_.]", file_name)[-2]
    output_folder = Path(os.path.join('scratch',f'{campaign}_{store}_{n_groups}_{get_timestamp()}_{generate_random_unique_id()}'))
    output_folder.mkdir(parents=True,exist_ok=True)
    return output_folder

def demo_grouping(output_folder, file_name, n_groups, splits, avg_tol, size_tol):
#def demo_grouping():
    """
    This function use clustering to  generate test control group
    :param file_name: z3 input file in json format
    :param n_groups: to set number of cluster
    :param splits: dict with split ratio i.e {'control': 0.1, 'test': 0.9}
    :param avg_tol: how much can the subset scores averages differ by
    :param size_tol: how much can subset sizes differ by, beyond what is expected from the splits
    """
    assets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ASSETS_DIR_RELPATH)
    #sample_data_path = os.path.join(assets_dir, r'jyc_circlek.json')


    #assets_dir = r'D:\work\project\cac\z3_explorations\assets'
    sample_data_path = os.path.join(assets_dir, file_name)
    print("Will process data from %s." % (sample_data_path,))
    campaign, store = get_campaign_info(file_name)
    with open(sample_data_path) as f:
        sample_data = json.loads(f.read())
    zipcode_to_latlong = ots.load_zipcodes()

    # configs
    #n_groups = 300
    #splits = {'control': 0.20, 'test': 0.80}
    #avg_tol = 8
    #size_tol = 100

    campaign, store = get_campaign_info(file_name)

    email_notification(output_folder, campaign, store, n_groups, avg_tol, size_tol)

    # output file names
    cluster_viz_file = os.path.join(output_folder, r'demo_%d_groups_%d_avg_tol_%d_size_tol.png' % (n_groups, avg_tol, size_tol, ) )
    set_splitter_output_file = os.path.join(output_folder, r'demo_splits_with_%d_groups_%d_avg_tol_%d_size_tol.xlsx' % (n_groups, avg_tol, size_tol, ) )
    cluster_expansion_to_store = os.path.join(output_folder, r'demo_%d_groups_%d_avg_tol_%d_size_tol_expansion_to_stores.json' % (n_groups, avg_tol, size_tol, ) )
    final_sets_viz_file = os.path.join(output_folder, r'demo_splits_with_stores_%d_groups_%d_avg_tol_%d_size_tol.png' % (n_groups, avg_tol, size_tol, ) )


    # execute
    # group stores first, because the original data is too big
    groups, store_centroids = ots.group_stores(sample_data, zipcode_to_latlong, n_groups=n_groups,
                                               plot_file=cluster_viz_file, distance_metric='euclidean')
    # create merged dict that will then be passed to the splitter
    merged_data, cluster_name_to_store_map = ots.merge_stores(sample_data, groups)
    # split - but remember to set use_size_info=True
    subsets_with_name = ots.set_splitter(merged_data, splits=splits, results_file=set_splitter_output_file,
                                                        avg_tol=avg_tol, size_tol=size_tol, use_size_info=True)

    # expand back the groups to store names
    store_splits = ots.expand_groups_into_stores(subsets_with_name, cluster_name_to_store_map)

    pd.DataFrame.from_dict(store_splits, orient ='index').transpose().to_json(cluster_expansion_to_store)
    # optional: plot the subset assignment, remember red is for control
    ots.plot_store_sets(store_splits, store_centroids, plot_file=final_sets_viz_file,
                        control_key='control')

    email_group_info(output_folder, campaign, store, n_groups, avg_tol, size_tol)



if __name__ == "__main__":
    file_name = sys.argv[1]
    #file_name = "mtwdw_doritos_7e.json"
    n_groups = int(sys.argv[2])
    #n_groups = 300
    split_json = sys.argv[3]
    #split_json = '{"control":0.20,"test":0.80}'
    #print("passed json", split_json)
    splits = json.loads(split_json)
    ##splits = sys.argv[3]
    avg_tol = int(sys.argv[4])
    #avg_tol = 5
    size_tol = int(sys.argv[5])
    #size_tol = 100
    output_folder = create_output_folder(file_name, n_groups, avg_tol, size_tol)
    atexit.register(email_notification_exit, output_folder, file_name, n_groups, avg_tol, size_tol)
    #atexit.register(email_notification_exit, n_groups)
    try:
        demo_grouping(output_folder, file_name, n_groups, splits, avg_tol, size_tol)
        #raise Exception("testing exception")
    except Exception as e:
        exception_msg = traceback.format_exc()
        raise e

    #print(os.getlogin())
    #time.sleep(60)
    success = True
    #cluster = sys.argv[1]
    #file_name = sys.argv[2]
    #splits = json.loads(sys.argv[3])
    #avg_tol = int(sys.argv[4])
    #size_tol = int(sys.argv[5])
    #n_groups = int(sys.argv[6])
    #results_file = sys.argv[7]
    ##demo_grouping(file_name, n_groups, splits, avg_tol, size_tol)
    #run_splitter(cluster, file_name, splits, avg_tol, size_tol, n_groups=None, results_file=None)
