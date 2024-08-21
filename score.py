import argparse
from scoremodel import score_model

parser = argparse.ArgumentParser()
parser.add_argument('--aspect', required=True, type=str, help='ALL,UTMOS,NISQA,CER or SECS')
parser.add_argument('--mode',  type=str, default="predict_dir",help='either predict_file or predict_dir')
parser.add_argument('--output_dir', type=str,default='score_result', help='folder to ouput results.csv')
parser.add_argument('--deg', type=str, help='path to speech file')
parser.add_argument('--data_dir', type=str, help='folder with speech files')

"""NISQA"""
parser.add_argument('--pretrained_model',default="NISQA/weights/nisqa_tts.tar", type=str, help='file name of pretrained model (must be in current working folder)')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for pytorchs dataloader')
parser.add_argument('--bs', type=int, default=1, help='batch size for predicting')
parser.add_argument('--ms_channel', type=int, help='audio channel in case of stereo file')

"""SECS"""
parser.add_argument('--refer', type=str, help='file path of the reference audio')

"""CER"""
parser.add_argument('--model_size', type=str,default='base', help='tiny, base,small,medium,large')


args = parser.parse_args()
args = vars(args)

if __name__ == '__main__':
    score=score_model(args)
    score.predict()
