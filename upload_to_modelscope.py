from modelscope.hub.api import HubApi
import os

# ====================== 修复后的参数（关键改 repo_type！）======================
USER_TOKEN = "ms-25091a3f-173e-48bb-9ed0-961f0d6821a9"
# 🔥 强烈建议：换成精简文件夹（只放 app.py、requirements.txt、模型文件）
LOCAL_FOLDER = r"D:\Study\ultralytics-main\garbage_detection"  # 可改成你的精简文件夹路径
REPO_NAMESPACE = "HanJQ20191226"
REPO_NAME = "123lajifenlei"
REPO_TYPE = "model"  # 改为支持的 model 类型（替代 application）
# ===========================================================================

# 初始化API并登录
api = HubApi()
api.login(USER_TOKEN)
repo_id = f"{REPO_NAMESPACE}/{REPO_NAME}"

def upload_all_files():
    if not os.path.exists(LOCAL_FOLDER):
        raise Exception(f"❌ 本地文件夹不存在！路径：{LOCAL_FOLDER}")

    uploaded_files = []
    failed_files = []
    print(f"📤 开始上传 → 仓库：{repo_id}（类型：{REPO_TYPE}）\n")

    # 遍历文件夹，只上传核心文件（排除无用的 docs、tests、examples 等）
    for root, dirs, files in os.walk(LOCAL_FOLDER):
        # 跳过无关文件夹（大幅减少上传文件数）
        if any(exclude in root for exclude in ["docs", "tests", "examples", ".github", "docker", "macros", "overrides"]):
            continue

        for file_name in files:
            # 只保留核心文件类型（可根据你的需求调整）
            allowed_ext = [".py", ".txt", ".yaml", ".yml", ".pt", ".pth", ".jpg", ".png"]
            if not any(file_name.endswith(ext) for ext in allowed_ext):
                continue

            local_file = os.path.join(root, file_name)
            repo_file_path = os.path.relpath(local_file, LOCAL_FOLDER)

            try:
                # 用支持的 repo_type=model 上传
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=repo_file_path,
                    repo_id=repo_id,
                    repo_type=REPO_TYPE,
                    commit_message=f"Upload {repo_file_path}",
                    disable_tqdm=True
                )
                uploaded_files.append(repo_file_path)
                print(f"✅ {repo_file_path}")
            except Exception as e:
                error_msg = str(e)
                if "large file" in error_msg.lower() or "lfs" in error_msg.lower():
                    error_msg += " → 运行 pip install git-lfs 后重试"
                failed_files.append(f"{repo_file_path}：{error_msg[:50]}...")
                print(f"❌ {repo_file_path}")

    # 输出总结
    print("\n" + "="*60)
    print(f"📊 上传总结：成功 {len(uploaded_files)} 个，失败 {len(failed_files)} 个")
    if failed_files:
        print(f"❌ 失败示例：{failed_files[0]}")
    print(f"\n📦 仓库地址：https://modelscope.cn/{repo_id}")
    print(f"👉 下一步：登录魔搭「创空间」→ 新建项目 → 选择该仓库部署Gradio！")

if __name__ == "__main__":
    try:
        upload_all_files()
    except Exception as e:
        print(f"\n❌ 上传异常：{str(e)}")