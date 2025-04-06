import time
import smtplib
import imghdr
from email.message import EmailMessage
import glob
import atexit
import random, string
import traceback
from pathlib import Path
import os
import run_splitter as rs

success = False
exception_msg = ""

def email_notifier(config_dict, state):
    EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')
    contacts = ['shubham.gupta@247.ai', 'vikram.gupta@247.ai', 'sahana.ka@247.ai']
    #contacts = ['shubham.gupta@247.ai']
    msg = EmailMessage()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = contacts
    user = (os.getlogin()).split('.')[0]

    if config_dict['mode'] == 'multi':
        store = ' , '.join([rs.get_retailer(file) for file in config_dict['z3_input_file']])
        avg_tol = " , ".join([str(_) for _ in config_dict['avg_tol']])
        size_tol = " , ".join([str(_) for _ in config_dict['size_tol']])
        if config_dict['cluster']:
            groups = " , ".join([str(_) for _ in config_dict['groups']])

    else:
        store = config_dict['store']
        avg_tol = config_dict['avg_tol']
        size_tol = config_dict['size_tol']
        if config_dict['cluster']:
            groups = config_dict['groups']


    if state == "terminated":
        if config_dict['cluster']:
            body = f"{user} z3 jobs has been terminated for {config_dict['campaign']} campaign for {config_dict['client']} for {config_dict['brand']} for {store} with these parameters: groups {groups} , avg_tol {avg_tol} and size_tol {size_tol} at this location {config_dict['output_folder']}, exception message is {exception_msg}"
        else:
            body = f"{user} z3 jobs has been terminated for {config_dict['campaign']} campaign for {config_dict['client']} for {config_dict['brand']} for {store} with these parameters: avg_tol {avg_tol} and size_tol {size_tol} at this location {config_dict['output_folder']}, exception message is {exception_msg}"

        if not success:
            msg['Subject'] = f"{user} z3 jobs for {config_dict['campaign']} campaign {store} store has been terminated"

            with open(os.path.join(config_dict['output_folder'], 'failure_msg.txt'), 'w') as f:
                f.write(body)
            msg.set_content(body)
        else:
            with open(os.path.join(config_dict['output_folder'], 'success_msg.txt'), 'w') as f:
                f.write(body)

    elif state == "started":
        msg['Subject'] = f"{config_dict['campaign']} z3 jobs for {store}"
        if config_dict['cluster']:
            body = f"{user} z3 jobs has been started for {config_dict['campaign']} campaign for {store} with these parameters: groups {groups}, avg_tol {avg_tol} and size_tol {size_tol}"
        else:
            body = f"{user} z3 jobs has been started for {config_dict['campaign']} campaign for {store} with these parameters: avg_tol {avg_tol} and size_tol {size_tol}"
        msg.set_content(body)


    elif state == "finished":
        msg['Subject'] = f"{config_dict['campaign']} z3 jobs for {store}"
        if config_dict['cluster']:
            body = f"{user} here are the split details generated for {config_dict['campaign']} campaign for {store} from these parameters: groups {groups}, avg_tol {avg_tol} and size_tol {size_tol}"
        else:
            body = f"{user} here are the split details generated for {config_dict['campaign']} campaign for {store} from these parameters: avg_tol {avg_tol} and size_tol {size_tol}"
        msg.set_content(body)
        files = glob.glob(os.path.join(config_dict['output_folder'], r'*split*'))

        for file in files:
            #filename = file.split('\\')[-1]
            filename = Path(file).name
            #filetype = file.split('\\')[-1].split('_')[-1].split('.', 1)[-1]
            filetype = Path(file).name.split('.')[-1]

            if filetype == 'png':
                with open(file, 'rb') as f:
                    file_data = f.read()
                    image_type = imghdr.what(f.name)
                msg.add_attachment(file_data, maintype='image', subtype=image_type, filename=filename)


            else:
                with open(file, 'rb') as f:
                    file_data = f.read()
                msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=filename)

    with smtplib.SMTP('mail.svc.cloud.247-inc.net', port=25) as smtp:
        smtp.send_message(msg)