import subprocess
import os


def call_stroop_exe():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 stroop.exe 的路径
    stroop_exe_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'stroop', 'stroop.exe'))

    # 检查文件是否存在
    if not os.path.isfile(stroop_exe_path):
        raise FileNotFoundError(f"Could not find oddball.exe at {stroop_exe_path}")

    # 调用 oddball.exe
    result = subprocess.run([stroop_exe_path], capture_output=True, text=True)

    # 输出结果
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


# 在需要的时候调用函数
if __name__ == "__main__":
    call_stroop_exe()
