from train import fine_tuning
from wettbewerb import load_references, save_predictions
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train given pretrained Model')
    parser.add_argument('--train_dir', action='store',type=str,default='../train/')
    parser.add_argument('--model_name', action='store',type=str,default='model_5.h5')
    parser.add_argument('--from_scratch', action='store',type=bool,default=False)
    args = parser.parse_args()
    
    ecg_leads,ecg_labels,fs,ecg_names = load_references(args.train_dir) # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
    
    start_time = time.time()
    model_fine_tuned = fine_tuning(ecg_leads,fs,ecg_names,model_name=args.model_name,from_scratch=args.from_scratch)
    train_time = time.time()-start_time
    
    print("Runtime",train_time,"s")
