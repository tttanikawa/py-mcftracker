import shutil
import os
import glob
import cv2

training_data_path = '/home/dmitriy.khvan/pytorch-beginner/training/'
validation_data_path = '/home/dmitriy.khvan/pytorch-beginner/validation/'

training_data_dest = '/home/dmitriy.khvan/deep-person-reid/reid-data/bepro/bounding_box_train'
query_data_dest = '/home/dmitriy.khvan/deep-person-reid/reid-data/bepro/query'
test_data_dest = '/home/dmitriy.khvan/deep-person-reid/reid-data/bepro/test'

copy_data_path = '/home/dmitriy.khvan/deep-person-reid/reid-data/bepro_tmp'

train_dir_list = os.listdir(training_data_path)
valid_dir_list = os.listdir(validation_data_path)
tmp_dir_list = os.listdir(copy_data_path)

def copy_files():
    count = 0

    for d in train_dir_list:
        folder_indices = [0]

        copy_dir = os.path.join(copy_data_path, d)
        os.mkdir(copy_dir)

        for fi in folder_indices:
            path2dir = os.path.join(training_data_path, d, str(fi))
            
            if os.path.isdir(path2dir):
                # pid_list = os.listdir(path2dir)
                pid_list = ['0', '1']

                for p in pid_list:
                    copy_dir_low = os.path.join(copy_dir, p)
                    os.mkdir(copy_dir_low)

                    path2pid = os.path.join(path2dir, p)

                    if os.path.isdir(path2pid):
                        for filename in os.listdir(path2pid):
                            path2file = os.path.join(path2pid, filename)

                            imgidx = int(filename.split('.')[0])
                            img_indices = [0, 10, 15, 20, 25, 29]

                            if imgidx in img_indices:
                                print (path2file)
                                image = cv2.imread(path2file)
                                
                                if image.shape[0] < 110 and image.shape[1] < 40:
                                    continue
            
                                path2cp_dest = os.path.join(copy_dir_low, filename)

                                image_res = cv2.resize(image, (64,128))
                                cv2.imwrite(path2cp_dest, image_res)
                                count = count + 1

    print (count)
    

def create_test_dataset(pid):
    pid = pid +1 

    count_q = 0
    count_t = 0

    for d in valid_dir_list:

        folder_indices = [0,1,2,3]

        for fi in folder_indices:
            path2dir = os.path.join(validation_data_path, d, str(fi))

            if os.path.isdir(path2dir):
                pid_list = os.listdir(path2dir)

                for p in pid_list:
                    path2pid = os.path.join(path2dir, p)
                    
                    pid = pid + 1
                    for filename in os.listdir(path2pid):
                        path2file = os.path.join(path2pid, filename)

                        imgidx = int(filename.split('.')[0])
                        img_indices = [5, 10, 15, 20, 25, 29]

                        image = cv2.imread(path2file)
                        
                        if imgidx == 0:
                            #copy to query
                            new_filename = '%d_%s_%d.jpg' % (pid, '0', imgidx)
                            path2dest = os.path.join(query_data_dest, new_filename)
                            count_q = count_q + 1
                        elif imgidx in img_indices:
                            #copy to test
                            new_filename = '%d_%s_%d.jpg' % (pid, '1', imgidx)
                            path2dest = os.path.join(test_data_dest, new_filename)
                            count_t = count_t + 1
                        else:
                            continue

                        print (path2dest)

                        image_res = cv2.resize(image, (64,128))
                        cv2.imwrite(path2dest, image_res)

    print (count_q)
    print (count_t)

def create_training_dataset():
    pid = 1
    
    for d in tmp_dir_list:
        
        path2dir = os.path.join(copy_data_path, d)
        pid_list = os.listdir(path2dir)

        print (d)
        print (pid_list)
        
        for p in pid_list:
            path2pid = os.path.join(path2dir, p)
            print (path2pid)
            pid = pid + 1
            fcount = 0

            for filename in os.listdir(path2pid):
                path2file = os.path.join(path2pid, filename)
                print (path2file)

                image = cv2.imread(path2file)

                new_filename = '%d_%s_%d.jpg' % (pid, '1', fcount)
                path2dest = os.path.join(training_data_dest, new_filename)

                # print (path2dest)
                cv2.imwrite(path2dest, image)

                fcount = fcount + 1

    return pid

if __name__ == "__main__":
    # copy_files()
    pid = create_training_dataset()
    create_test_dataset(pid)
