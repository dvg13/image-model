import os
import shutil
import sys

def flatten_lfw(lfw_dir,output_dir,rename=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for dirs,subs,files in os.walk(lfw_dir):
        for sub in subs:
            for dirs,subs,files in os.walk(os.path.join(lfw_dir,sub)):
                for f in files:
                    if f.endswith(".jpg"):
                        shutil.copy(os.path.join(lfw_dir,sub,f),os.path.join(output_dir,f))

def main(lfw_dir,output_dir,rename=False):
    flatten_lfw(lfw_dir,output_dir,rename)

if __name__ == "__main__":
    lfw_dir,output_dir = sys.argv[1:3]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    main(lfw_dir,output_dir)
