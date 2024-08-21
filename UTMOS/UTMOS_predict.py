import torch
import librosa
import os
import csv

def score_file(file_path:str,output_dir:str):
    output_file = os.path.join(output_dir, 'UTMOS.csv')
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'score'])  # 写入表头

        wave, sr = librosa.load(file_path, sr=None, mono=True)
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        score = predictor(torch.from_numpy(wave).unsqueeze(0), sr)
        score_value = score.detach().item()
        print(score_value)

        writer.writerow([file_path, score_value])
    return score

def score_directory(dir_path:str, output_dir:str):
    audio_folder = dir_path
    wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

    # 加载预训练模型
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    folder_name = os.path.basename(os.path.normpath(dir_path))
    output_file = os.path.join(output_dir, f'{folder_name}_UTMOS.csv')
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'score'])  # 写入表头
    # 遍历每个wav文件并打分
        for wav_file in wav_files:
            file_path = os.path.join(audio_folder, wav_file)
            wave, sr = librosa.load(file_path, sr=None, mono=True)
            score = predictor(torch.from_numpy(wave).unsqueeze(0), sr)
            score_value = score.detach().item()
            print(f"{wav_file}: {score_value}")

            writer.writerow([wav_file, score_value])

if __name__ == '__main__':
    # score_file()
    score_directory()