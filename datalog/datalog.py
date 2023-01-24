import time
import os
import datetime

def writing_txt_file(experiment="",argument="",destination_folder="",file_name=""):
    with open(os.path.join(destination_folder,file_name) + ".txt", 'w') as file:
        e = datetime.datetime.now()
        file.write(f"Experiment: {experiment}\n")
        file.write(argument)
        file.write("\n")
        file.write(e.strftime("%Y-%m-%d %H:%M:%S"))