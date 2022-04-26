import numpy as np
import csv
import argparse

label_dict = {
        "Drink" : 0,
        "Jump"  : 1,
        "Pick"  : 2,
        "Pour"  : 3,
        "Push"  : 4,
        "Run"   : 5,
        "Sit"   : 6,
        "Stand" : 7,
        "Turn"  : 8,
        "Walk"  : 9,
        "Wave"  : 1,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
    parser.add_argument('--csv', default='track1_pred.csv', help="pred csv path")
    args = parser.parse_args()
    
    rows = []
    with open(args.csv, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    num_videos = 0.0
    match = 0.0

    for row in rows:
        pred = int(row[2])
        gt = label_dict[row[1].split("/")[0]]
        match += pred == gt
        num_videos += 1
        # print(row[1], pred, gt)

    acc = match / num_videos

    print(f"Accuracy : {acc * 100}")

