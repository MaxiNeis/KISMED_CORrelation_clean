from train import fine_tuning
from wettbewerb import load_references, save_predictions
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train given pretrained Model')
    parser.add_argument('--train_dir', action='store',type=str,default='C:\\Users\\maxim\\TU Darmstadt\\Wettbewerb KISMED\\KISMED_CORrelation_clean\\training_KISMED')
    parser.add_argument('--model_name', action='store',type=str,default='model_5.h5')                           # Ausgangsmodell für Finetuning
    parser.add_argument('--from_scratch', action='store',type=bool,default=False)                                # boolean, ob ein Modell gefinetuned werden soll (False) oder ein neues Modell von Grundauf erstellt werden soll (True) 
    parser.add_argument('--new_model_name', action='store',type=str,default='Transformer_Encoder_finetuned.h5') # Name des neuen Modells nach Finetuning oder Trainieren von Grundauf
    args = parser.parse_args()
    
    ecg_leads,ecg_labels,fs,ecg_names = load_references(args.train_dir) # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name
        
    start_time = time.time()
    model_fine_tuned = fine_tuning(ecg_leads, ecg_labels, fs, ecg_names, model_name=args.model_name, new_model_name=args.new_model_name, from_scratch=args.from_scratch)
    train_time = time.time()-start_time
    
    print("Runtime",train_time,"s")
