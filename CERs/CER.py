import whisper
import os
import csv

def transcribe(dir_path:str,model,output_dir):
    model = whisper.load_model(model)

    wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
    text_files = {os.path.splitext(f)[0]: f for f in os.listdir(dir_path) if f.endswith('.txt')}

    folder_name = os.path.basename(os.path.normpath(dir_path))
    output_file = os.path.join(output_dir, f'{folder_name}_CER.csv')

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'CER'])

        for wav_file in wav_files:
            file_path=os.path.join(dir_path,wav_file)
            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)

            # decode the audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)

            prefix = os.path.splitext(wav_file)[0]
            if prefix in text_files:
                with open(os.path.join(dir_path, text_files[prefix]), 'r', encoding='UTF-8') as txt_file:
                    reference_text = txt_file.read().strip()
                cer = get_cer(result.text, reference_text)
                writer.writerow([wav_file, cer])
                print(f"the CER of {wav_file} is {cer}")

            else:
                writer.writerow([wav_file, "None"])
                print(f"No matching text file for {wav_file}")

def edit_distance(str1, str2):
    """计算两个字符串之间的编辑距离。
    Args:
        str1: 字符串1。
        str2: 字符串2。
    Returns:
        dist: 编辑距离。
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    dist = matrix[len(str1)][len(str2)]
    return dist

def get_cer(src, trg):
    """把源字符串src修改成目标字符串trg的字符错误率。
    Args:
        src: 源字符串。
        trg: 目标字符串。
    Returns:
        cer: 字符错误率。
    """
    dist = edit_distance(src, trg)
    cer = dist / len(trg)
    return cer
