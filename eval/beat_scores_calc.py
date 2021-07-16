#!/usr/bin/env python
# encoding: utf-8
"""
calc beat score of files
copyright: www.mgtv.com
"""

import os
import sys
import argparse
import numpy as np
import traceback
import beat_evaluation_toolbox as be

        
def calc_beat_score_of_file(annotation_file, beat_file):
    #check input params        
    if os.path.exists(annotation_file) == False:
        print("failed! annotation_file:%s not exist\n" % (annotation_file))
        return False, 0.0
        
    if os.path.exists(beat_file) == False:
        print("failed! beat_file:%s not exist\n" % (beat_file))
        return False, 0.0
        
        
    data_annotation = np.loadtxt(annotation_file, usecols=(0))
    data_annotation = np.expand_dims(data_annotation, axis=0)

        
    data_beat = np.loadtxt(beat_file, usecols=(0))
    data_beat = np.expand_dims(data_beat, axis=0)
    
    R = be.evaluate_db(data_annotation, data_beat, 'all', doCI=False)
    
    #输出结果
    print(R['scores'])
    pscore = R['scores']['pScore'][0]
    f_measure = R['scores']['fMeasure'][0]
    
    aml_c = R['scores']['amlC'][0]
    aml_t = R['scores']['amlT'][0]
    cml_c = R['scores']['cmlC'][0]
    cml_t = R['scores']['cmlT'][0]
    
    cem_acc = R['scores']['cemgilAcc'][0]
    
    total_score = (aml_c + cem_acc + cml_c  + f_measure + pscore + cml_t + aml_t) / 7
    print("[%s] score:%.4f"%(beat_file, total_score))
    return True, total_score



def calc_avg_score_of_files(annotation_files_dir, beat_files_dir, file_extension):
    #check input params        
    if os.path.exists(annotation_files_dir) == False:
        print("failed! annotation_files_dir:%s not exist\n" % (annotation_files_dir))
        return False, 0.0
        
    if os.path.exists(beat_files_dir) == False:
        print("failed! beat_files_dir:%s not exist\n" % (beat_files_dir))
        return False, 0.0
        
    if not annotation_files_dir.endswith("/"):
        annotation_files_dir += "/"

    if not beat_files_dir.endswith("/"):
        beat_files_dir += "/"

    annotation_files_url = [f for f in os.listdir(annotation_files_dir) if f.endswith((file_extension))]
    nb_annotation_files = len(annotation_files_url)

    beat_files_url = [f for f in os.listdir(beat_files_dir) if f.endswith((file_extension))]
    nb_beat_files = len(beat_files_url)
    
    if nb_annotation_files != nb_beat_files or nb_annotation_files == 0:
        print("failed! annotation files num:%d  beat files num:%d\n" % (nb_annotation_files, nb_beat_files))
        return False, 0.0
    
    sum_score = 0.0
    for i in range(nb_annotation_files):
        annotation_file = annotation_files_dir + annotation_files_url[i]
        beat_file = beat_files_dir + annotation_files_url[i]
        
        if os.path.exists(beat_file) == False:
            print("failed! beat file:%s not exist\n" % (beat_file))
            return False, 0.0
            
        ret, score = calc_beat_score_of_file(annotation_file, beat_file)
        if ret == False:
            print("failed! calc_beat_score_of_file failed for file:%s\n" % (beat_file))
            return False, 0.0
            
        sum_score = sum_score + score
    
    avg_score = sum_score / nb_annotation_files
   
    return True, avg_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="calc avg score of beat(downbeat) files")
    parser.add_argument("--annotation_files_dir", required=True, help="Path to input annotation files dir", default="")
    parser.add_argument("--beat_files_dir", required=True, help="Path to input beats files dir", default="")
    parser.add_argument("--file_extension", required=True, help="File ext, beat or downbeat", default="")
    
    # 获得工作目录，程序模块名称，并切换工作目录
    s_work_path, s_module_name = os.path.split(os.path.abspath(sys.argv[0]))
    print(s_work_path, s_module_name)
    os.chdir(s_work_path)
    
    try:
        args = parser.parse_args()
        ret, score = calc_avg_score_of_files(args.annotation_files_dir, args.beat_files_dir, args.file_extension)
        print("Final score:%.4f" % score)
            
    except Exception as e:        
        traceback.print_exc()
        print("Exception running beat_score_calc: [%s]" % (str(e)))
        ret = False
        
    if ret == True:
        sys.exit(0)
    else:
        sys.exit(1)
