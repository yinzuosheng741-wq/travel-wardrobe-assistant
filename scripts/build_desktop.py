"""Build a Windows directory distribution, without developer data or credentials."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.package_notices import copy_notices  # noqa: E402
from stylemate.branding import APP_NAME, ARCHIVE_STEM  # noqa: E402


def copy_user_guides(output: Path):
    """Preserve guide/screenshot relative paths in the extracted application."""
    docs = output / "docs"
    (docs / "guides").mkdir(parents=True, exist_ok=True)
    for name in ("DESKTOP.md", "SETUP.md"):
        shutil.copyfile(ROOT / "docs/guides" / name, docs / "guides" / name)
    (docs / "assets").mkdir(exist_ok=True)
    for name in ("service-setup.png", "weather-setup.png"):
        shutil.copyfile(ROOT / "docs/assets" / name, docs / "assets" / name)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    output = ROOT / "dist/native" / APP_NAME
    if any((output / name).exists() for name in ("data/stylemate.db", "data/config/services.env", ".env")):
        raise RuntimeError("分发目录已有个人数据，请先备份并移走旧分发目录。")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
               "--contents-directory", ".", "--name", APP_NAME, "--paths", str(ROOT),
               "--distpath", str(ROOT / "dist/native"), "--workpath", str(ROOT / "dist/native-build"),
               "--specpath", str(ROOT / "dist/native-build")]
    for name in ("assets/demo", "stylemate/prompts"):
        command.extend(["--add-data", f"{ROOT / name};{name}"])
    for name in ("records.jsonl", "sources.json"):
        command.extend(["--add-data", f"{ROOT / 'data/knowledge' / name};data/knowledge"])
    for folder in ("stylemate/skillpacks", "stylemate/constraints"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in {".yaml", ".md"}:
                destination = path.parent.relative_to(ROOT).as_posix()
                command.extend(["--add-data", f"{path};{destination}"])
    command.extend(["--add-data", f"{ROOT / '.env.example'};."])
    for name in ("langchain", "langchain-core", "langchain-openai", "langgraph", "langgraph-checkpoint",
                 "langgraph-prebuilt", "langgraph-sdk", "dashscope"):
        command.extend(["--copy-metadata", name])
    command.extend(["--collect-submodules", "langgraph", "--hidden-import", "tiktoken_ext.openai_public"])
    command.extend(["--collect-all", "milvus_lite", "--collect-all", "faiss", "--collect-all", "pyarrow"])
    for name in ("streamlit", "chromadb", "matplotlib", "scipy", "pandas", "torch", "tensorflow",
                 "PyQt5", "PyQt6", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"):
        command.extend(["--exclude-module", name])
    command.append(str(ROOT / "desktop.py"))
    subprocess.run(command, cwd=ROOT, check=True)
    (output / "使用说明.txt").write_text(
        f"双击 {APP_NAME}.exe 启动。请保留整个目录，不要只复制 exe。\n"
        "首次启动可配置模型、天气与定位，也可以选择暂不配置。\n"
        "以后在 设置 > 服务状态 > 配置模型、天气与定位 中修改并测试连接。\n"
        "衣橱、配置和界面状态都保存在本目录 data/ 下；退出后可整个文件夹迁移。\n"
        "data/config/services.env 保存私有密钥，不要分享自己使用过的整个文件夹。\n"
        "公开分发请使用项目生成的不含个人数据的 ZIP。\n", encoding="utf-8")
    copy_user_guides(output)
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")
    copy_notices(output)
    forbidden = [path for path in output.rglob("*") if path.is_file() and
                 (path.name == ".env" or path.suffix in {".db", ".ini"} or
                  "config" in path.relative_to(output).parts and path.suffix == ".env")]
    if forbidden:
        raise RuntimeError("公开分发目录包含运行数据或私有配置，已停止打包。")
    archive = shutil.make_archive(str(ROOT / "dist" / ARCHIVE_STEM), "zip", output.parent, output.name)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise RuntimeError("ZIP 完整性校验失败。")
    print(archive)
    print(output)


if __name__ == "__main__":
    main()
