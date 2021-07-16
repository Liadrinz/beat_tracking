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

def calc_beat_score_of_file(args):
    #check input params        
    if os.path.exists(args.annotation_file) == False:
        print("failed! annotation_file:%s not exist\n" % (args.annotation_file))
        return False, 0.0
        
    if os.path.exists(args.beat_file) == False:
        print("failed! beat_file:%s not exist\n" % (args.beat_file))
        return False, 0.0
        
        
    data_annotation = np.loadtxt(args.annotation_file, usecols=(0))
    data_annotation = np.expand_dims(data_annotation, axis=0)

        
    data_beat = np.loadtxt(args.beat_file, usecols=(0))
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
    print("Final score:%.4f"%total_score)
    return True, total_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="calc score of beat file")
    parser.add_argument("--annotation_file", required=True, help="Path to input annotation file", default="")
    parser.add_argument("--beat_file", required=True, help="Path to input beats file", default="")
    
    try:
        args = parser.parse_args()
        ret, score = calc_beat_score_of_file(args)

            
    except Exception as e:        
        traceback.print_exc()
        print("Exception running beat_score_calc: [%s]" % (str(e)))
        ret = False
        
    if ret == True:
        sys.exit(0)
    else:
        sys.exit(1)
