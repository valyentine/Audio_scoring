import os
import pandas as pd
import csv

def merge_csv_files(dir_path: str, output_dir: str):
    # Extract the folder name from the directory path
    folder_name = os.path.basename(os.path.normpath(dir_path))

    utmos_csv_path = os.path.join(output_dir, f'{folder_name}_UTMOS.csv')
    nisqa_csv_path = os.path.join(output_dir, f'{folder_name}_NISQA_results.csv')
    secs_csv_path = os.path.join(output_dir, f'{folder_name}_SECS.csv')
    cer_csv_path = os.path.join(output_dir, f'{folder_name}_CER.csv')


    utmos_df = pd.read_csv(utmos_csv_path, encoding='GBk')
    nisqa_df = pd.read_csv(nisqa_csv_path, encoding='UTF-8')
    secs_df = pd.read_csv(secs_csv_path, encoding='GBk')
    cer_df = pd.read_csv(cer_csv_path, encoding='GBK')




    filenames = utmos_df['filename'].tolist()
    UTMOSs=utmos_df['score'].tolist()
    NISQAs=nisqa_df['mos_pred'].tolist()
    SECSs=secs_df['Similarity'].tolist()
    CERs=cer_df['CER'].tolist()

    # List all CSV files in the directory that start with the folder name as a prefix
    # csv_files = [f for f in os.listdir(dir_path) if f.startswith(folder_name) and f.endswith('.csv')]

    output_file = os.path.join(output_dir, f'{folder_name}_ALL.csv')

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'UTMOS', 'NISQA', 'SECS','CER'])  # Write the header
        for filename,utmos,nisqa,secs,cer in zip(filenames,UTMOSs,NISQAs,SECSs,CERs):
            writer.writerow([filename, utmos, nisqa, secs,cer])

    os.remove(utmos_csv_path)
    os.remove(nisqa_csv_path)
    os.remove(secs_csv_path)
    os.remove(cer_csv_path)
if __name__ == '__main__':
    merge_csv_files("audio/shoulinrui/exp_shoulinrui.m4a/shoulinrui.m4a_0000063040_0000325440.wav", 'score_result')