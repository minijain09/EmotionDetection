import os

def rename(directory):
    for name in os.listdir(directory):
        print(name)
        s=name.split('_')
        newname=s[0]+"_action_detect_cropped.jpg"
        os.rename(os.path.join(directory,name), 
                  os.path.join(directory,newname))


path ="C:\\Users\\minij\\Desktop\\Emotion Detection\\Frames2\\ghajini_drama\\Detected cropped"
rename(path)
