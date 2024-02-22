import os
import glob
import random
import shutil

def main():
    data_dir = "D:/pilot_test_dataset_v1"
    img_dir = "%s/tmp_images" % (data_dir)
    lbl_dir = "%s/labels" % (data_dir)
    template_dir = "D:/test_template_directory/labels"
    
    train_img_dir = "%s/images/train" % (data_dir)
    train_lbl_dir = "%s/labels/train" % (data_dir)
    valid_img_dir = "%s/images/valid" % (data_dir)
    valid_lbl_dir = "%s/labels/valid" % (data_dir)
    
    # NOTE: N images in directory
    img_list = [] # Size = N
    template_lbl = [] # Size = 5
    new_lbl = [] # Size = N
    
    print("Gathering images...")
    
    # Get image names
    for img in glob.glob("%s/*.jpg" % (img_dir)):
        img_list.append(os.path.splitext(os.path.basename(img))[0])
        
    print("Gathering template labels...")
        
    # Get template files w/ bounding box coordinates
    for tmp in glob.glob("%s/*.txt" % (template_dir)):
        template_lbl.append(os.path.basename(tmp))
    
    print("Creating files...")
    
    for i in range(len(img_list)):
        aug_comb = img_list[i][len(img_list[i])-6:]
        with open ("%s/%s.txt" % (lbl_dir, img_list[i]), 'w') as output_file:
        
            if (aug_comb[0] == '1'):
                with open("%s/size1.txt" % (template_dir), 'r') as input_file:
                    for line in input_file:
                        output_file.write(line)
                        
            if (aug_comb[0] == '2'):
                with open("%s/size2.txt" % (template_dir), 'r') as input_file:
                    for line in input_file:
                        output_file.write(line)
                        
            if (aug_comb[0] == '3'):
                with open("%s/size3.txt" % (template_dir), 'r') as input_file:
                    for line in input_file:
                        output_file.write(line)
                        
            if (aug_comb[0] == '4'):
                with open("%s/size1.txt" % (template_dir), 'r') as input_file:
                    for line in input_file:
                        output_file.write(line)
                        
            if (aug_comb[0] == '5'):
                with open("%s/size5.txt" % (template_dir), 'r') as input_file:
                    for line in input_file:
                        output_file.write(line)
                        
    print("Image Annotation Complete!")
    
    print("==========")
    
    num_train = int(round((len(img_list) * 0.8), -2))
    num_valid = len(img_list) - num_train
    
    train_idx = random.sample(range(len(img_list)), num_train)
    
    print("Reallocating training set...")
    
    for i in train_idx:
        img_to_move = "%s.jpg" % (img_list[i])
        shutil.move("%s/%s" % (img_dir, img_to_move), "%s/%s" % (train_img_dir, img_to_move))
        
    print("Reallocating validation set...")
        
    for img in glob.glob("%s/*.jpg" % (img_dir)):
        img_to_move = os.path.basename(img)
        shutil.move("%s/%s" % (img_dir, img_to_move), "%s/%s" % (valid_img_dir, img_to_move))
    
    print("Image Reallocation Complete!")
    
    # RELOCATE LABELS HERE

if __name__ == "__main__":
    main()