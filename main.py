from option import Option
import subprocess
if __name__ == "__main__":
    option = Option()
    if option.train:
        subprocess.run(["python", "train.py"])
    else:
        subprocess.run(["python", "evaluation.py"])
