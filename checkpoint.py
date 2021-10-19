from helper import Helper
from config import Filenames
import os, sys
import torch as T


def save_checkpoint(checkpoint):
    Helper.check_folder(Filenames.backup_dir)
    file = get_backup_filename()
    Helper.remove_file(file)
    Helper.printline("=> Saving checkpoint")
    T.save(checkpoint, file)

def load_checkpoint(model, optimizer):
    Helper.printline("=> Loading checkpoint")
    file = get_backup_filename()
    if os.path.isfile(file) == False:
        sys.exit(f"!! Backup file does not exist for loading: {file}")
        
    checkpoint = T.load(file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch, model

def get_backup_filename():
    file = os.path.join(Filenames.backup_dir, Filenames.backup_file)
    return file
 