import javalang
import os
from utils import parse_statement, parse_source
from multiprocessing import Process


def generate_data(project):
    sub_dir = os.path.join(dir_name, project)
    err_num, success_num = 0, 0
    for file_name in os.listdir(sub_dir):
        try:
            with open(os.path.join(sub_dir, file_name), 'r') as f:
                source_code = f.readlines()
            new_source_code = parse_source(source_code)
            with open(os.path.join(out_dir, project + '_' + file_name), 'w') as f:
                f.writelines(new_source_code)
                success_num += 1
        except:
            err_num += 1
            print('error', project, file_name)
            pass
    print(err_num, success_num)


def main():
    process_list = []
    for i, project in enumerate(os.listdir(dir_name)):
        p = Process(target=generate_data, args=(project,))  # 实例化进程对象
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    print(GLOBAL_ERR)
    print(GLOBAL_SUCCESS)


if __name__ == '__main__':
    out_dir = 'dataset/code_embedding/'
    dir_name = 'dataset/raw_code/'
    GLOBAL_SUCCESS, GLOBAL_ERR = 0, 0
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        print('create directory', out_dir)
    main()
