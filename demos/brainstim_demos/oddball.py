import subprocess
import os


def call_oddball_exe():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 odmball.exe 的路径
    oddball_exe_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'oddball', 'oddball.exe'))

    # 检查文件是否存在
    if not os.path.isfile(oddball_exe_path):
        raise FileNotFoundError(f"Could not find oddball.exe at {oddball_exe_path}")

    # 调用 oddball.exe
    result = subprocess.run([oddball_exe_path], capture_output=True, text=True)

    # 输出结果
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


# 在需要的时候调用函数
if __name__ == "__main__":
    call_oddball_exe()
