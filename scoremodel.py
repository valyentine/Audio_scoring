from NISQA.nisqa.NISQA_model import nisqaModel
from UTMOS import UTMOS_predict
from SECS.SECS_predict import predictSECS
from fusion_csv import merge_csv_files
from CERs.CER import transcribe
class score_model():

    def __init__(self,args):
        self.args=args

    def predict(self):

        if self.args['aspect']=='NISQA':
            nisqa=nisqaModel(self.args)
            nisqa.predict()

        if self.args['aspect']=='UTMOS':
            if self.args['mode']=='predict_file':
                UTMOS_predict.score_file(self.args['deg'],self.args['output_dir'])
            else:
                UTMOS_predict.score_directory(self.args['data_dir'],self.args['output_dir'])

        if self.args['aspect']=='SECS':
            predictSECS(self.args['data_dir'],self.args['refer'],self.args['output_dir'])

        if self.args['aspect']=='CER':
            transcribe(self.args['data_dir'],self.args['model_size'],self.args['output_dir'])

        if self.args['aspect']=='ALL':
            nisqa = nisqaModel(self.args)
            nisqa.predict()

            if self.args['mode']=='predict_file':
                UTMOS_predict.score_file(self.args['deg'],self.args['output_dir'])
            else:
                UTMOS_predict.score_directory(self.args['data_dir'],self.args['output_dir'])

            predictSECS(self.args['data_dir'], self.args['refer'], self.args['output_dir'])

            transcribe(self.args['data_dir'],self.args['model_size'],self.args['output_dir'])

            merge_csv_files(self.args['data_dir'], self.args['output_dir'])

            print(f"csv file saved in {self.args['output_dir']}")
