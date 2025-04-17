'''2024-05-21
Input: [tissue, dk_struct]
Output: [lh, rh]
'''

import SimpleITK as sitk
import numpy as np
import os
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='UniTSeg Surface Preprocessing', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--input_dir', '-i', help='输入数据目录', required=True)
parser.add_argument('--output_dir', '-o', help='输出数据目录', required=True)
args = parser.parse_args()

def process_subject(subj_dir, output_dir):
    try:
        logger.info(f"处理受试者: {subj_dir}")
        
        # 读取输入文件
        tissue_path = os.path.join(subj_dir, 'tissue.nii.gz')
        dk_struct_path = os.path.join(subj_dir, 'dk-struct.nii.gz')
        
        if not os.path.exists(tissue_path) or not os.path.exists(dk_struct_path):
            logger.warning(f"缺少必要的输入文件: {subj_dir}")
            return
            
        tissue = sitk.GetArrayFromImage(sitk.ReadImage(tissue_path))
        dk_struct = sitk.GetArrayFromImage(sitk.ReadImage(dk_struct_path))
        T1W_nii = sitk.ReadImage(tissue_path)

        # 生成左右半球
        lh, rh = get_hemisphere(tissue, dk_struct)
        aseg = get_aseg(lh, rh, dk_struct)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存左半球图像
        lh_nii = sitk.GetImageFromArray(lh)
        lh_nii.SetOrigin(T1W_nii.GetOrigin())
        lh_nii.SetSpacing(T1W_nii.GetSpacing())
        lh_nii.SetDirection(T1W_nii.GetDirection())
        sitk.WriteImage(lh_nii, os.path.join(output_dir, 'lh.nii.gz'))

        # 保存右半球图像
        rh_nii = sitk.GetImageFromArray(rh)
        rh_nii.SetOrigin(T1W_nii.GetOrigin())
        rh_nii.SetSpacing(T1W_nii.GetSpacing())
        rh_nii.SetDirection(T1W_nii.GetDirection())
        sitk.WriteImage(rh_nii, os.path.join(output_dir, 'rh.nii.gz'))

        # 保存aseg图像
        aseg_nii = sitk.GetImageFromArray(aseg)
        aseg_nii.SetOrigin(T1W_nii.GetOrigin())
        aseg_nii.SetSpacing(T1W_nii.GetSpacing())
        aseg_nii.SetDirection(T1W_nii.GetDirection())
        sitk.WriteImage(aseg_nii, os.path.join(output_dir, 'aseg.nii.gz'))

        logger.info(f"成功处理: {subj_dir}")
        
    except Exception as e:
        logger.error(f"处理失败 {subj_dir}: {str(e)}")

def main():
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return
        
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有子文件夹
    for subj_id in os.listdir(input_dir):
        subj_dir = os.path.join(input_dir, subj_id)
        if os.path.isdir(subj_dir):
            subj_output_dir = os.path.join(output_dir, subj_id)
            process_subject(subj_dir, subj_output_dir)

if __name__ == "__main__":
    main()
    logger.info("UniTSeg Surface Preprocessing 完成") 