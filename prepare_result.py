import os 
import zipfile
import matplotlib.pyplot as plt

def compressFiles(model_ssd, model_name):
    # Name of the zip file to create
    output_file_name = os.path.basename(model_name)
    zip_filename = os.path.join('./', f'{output_file_name}_trainpackage.zip')
    
    labels_path = model_ssd.replace(os.path.basename(model_ssd), 'labels.txt')

    # Create the zip file
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(os.path.join('./', model_ssd))
        zipf.write(os.path.join('./', labels_path))

    return zip_filename

def prepare_training_result(model_name):
  files = os.listdir("models/{}".format(model_name))
  fileLst = []
  for file in files:
      if "mb" in file:
          fileLst.append(file)

  fileLst.sort(key=lambda x: int(x.split("-")[3]))

  x = []
  y = []

  for file in fileLst:
      file = file.split("-")
      loss = file[5]
      lossNum = loss.split(".pth")
      y.append(round(float(lossNum[0]), 3))
      x.append(file[3])

  bestChkpt = ""
  runOnce = True 
  prevLoss = 0.0
  for file in fileLst:
      origFile = file
      file = file.split("-")
      loss = file[5]
      lossNum = loss.split(".pth")
      if runOnce:
          prevLoss = lossNum
          runOnce = False
      else:
          if lossNum < prevLoss:
              prevLoss = lossNum
              bestChkpt = origFile #file[0] + file[1] + file[2] + file[3] + file[4] + file[5]

  model_ssd = 'models/{}/{}'.format(model_name, bestChkpt)
  print("Best Checkpoint: {}".format(model_ssd))
  plt.plot(x, y)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training result")
  plt.show()
  print('Compressing results...')
  print('Compressed in:', compressFiles(model_ssd, model_name))
