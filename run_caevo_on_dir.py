from os import listdir
from argparse import ArgumentParser
from os.path import isfile, join
import subprocess
import os


if __name__ == "__main__":
    parser = ArgumentParser(description='Run CAEVO')
    parser.add_argument("--input-dir", type=str, default="NYT_temp",
            help="the path to the file directory")
    parser.add_argument("--caevo-path", type=str, default="./caevo",
            help="the path to the caevo directory")
    parser.add_argument("--out-dir", type=str, default="NYT_xml",
            help="output directory, need to be created first")
    args = parser.parse_args()

    abs_input_dir = os.path.abspath(args.input_dir)
    abs_out_dir = os.path.abspath(args.out_dir)

    os.chdir(args.caevo_path)

    file_list = listdir(abs_input_dir)

    for idx, file in enumerate(file_list):
        if file.split('.')[-1] == 'txt':
            command = f"./runcaevoraw.sh {abs_input_dir}/{file}"

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
            output, error = process.communicate()

            error = error.decode("utf-8")

            new_file_name = f"{file}.info.xml"
            if isfile(f"{abs_input_dir}/{new_file_name}"):
                command = f"mv {abs_input_dir}/{new_file_name} {abs_out_dir}/"
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"{file} finished! {len(file_list)-idx-1} left")
            else:
                print(f"Somethings are wrong about {file}!")

