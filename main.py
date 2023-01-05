import os

if __name__ == '__main__':
    # clone YOLOv5 repository
    #os.system("!git clone https://github.com/ultralytics/yolov5")  # change directory
    #os.system("pip install - qr requirements.txt  # install dependencies (ignore errors)#start training")
    #os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.system("yolov5/python train.py --img 640 --batch 2 --epochs 1 --data 'LicensePlate_dataset.yaml' --cfg. /models/yolov5s.yaml --weights '' --name yolov5s_results --cache")
    os.system("load_ext tensorboard")
    os.system("tensorboard --logdir runs")