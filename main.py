import os

if __name__ == '__main__':
    # clone YOLOv5 repository
    #os.system("!git clone https://github.com/ultralytics/yolov5")  # change directory
    #os.system("pip install -qr requirements.txt  # install dependencies (ignore errors)#start training")
    #os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.system("python train.py --img 640 --batch 2 --epochs 1 --data 'LicensePlate_dataset.yaml' --weights yolov5s.pt --name yolov5s_results")
    os.system("load_ext tensorboard")
    os.system("tensorboard --logdir runs")