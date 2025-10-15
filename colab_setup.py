# ===============================
# 🚀 Colab Git & Env Auto-Setup
# ===============================

import os
import subprocess
from dotenv import load_dotenv

# 1. 📂 Load environment variables from .env file
ENV_PATH = "/content/.env"   # Change this path if needed

print("📁 [STEP 1] Checking for .env file...")
if not os.path.exists(ENV_PATH):
    raise FileNotFoundError(f"❌ .env file not found at {ENV_PATH}")
print("✅ .env file found")

print("🧪 [STEP 2] Loading environment variables...")
load_dotenv(ENV_PATH)
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_EMAIL = os.getenv("GITHUB_EMAIL")
GITHUB_PAT = os.getenv("GITHUB_PAT")
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL")
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME")

print(f"   ├─ GitHub User: {GITHUB_USERNAME}")
print(f"   ├─ Repo Name: {GITHUB_REPO_NAME}")
print(f"   └─ Repo URL: {GITHUB_REPO_URL}")

# 2. 🧰 Configure Git
print("\n🔧 [STEP 3] Configuring Git user info...")
subprocess.run(["git", "config", "--global", "user.name", GITHUB_USERNAME], check=True)
subprocess.run(["git", "config", "--global", "user.email", GITHUB_EMAIL], check=True)
print("✅ Git configuration completed")

# 3. 🔐 Setup .netrc for authentication
print("\n🔑 [STEP 4] Creating secure .netrc file for authentication...")
netrc_content = f"machine github.com login {GITHUB_USERNAME} password {GITHUB_PAT}\n"
with open(os.path.expanduser("~/.netrc"), "w") as f:
    f.write(netrc_content)
os.chmod(os.path.expanduser("~/.netrc"), 0o600)
print("✅ .netrc file created successfully")

# 4. 🌀 Clone or Pull repo
print("\n🌀 [STEP 5] Checking if repo already exists...")
if not os.path.exists(GITHUB_REPO_NAME):
    print(f"📥 Cloning repository: {GITHUB_REPO_URL}")
    subprocess.run(["git", "clone", GITHUB_REPO_URL, GITHUB_REPO_NAME], check=True)
    print("✅ Repository cloned successfully")
else:
    print(f"🔄 Repository already exists. Pulling latest changes in {GITHUB_REPO_NAME}...")
    os.chdir(GITHUB_REPO_NAME)
    subprocess.run(["git", "pull"], check=True)
    os.chdir("..")
    print("✅ Repository updated with latest changes")

# 5. 📦 Install uv (dependency manager) if needed
print("\n📦 [STEP 6] Checking for uv installation...")
try:
    subprocess.run(["uv", "--version"], check=True, stdout=subprocess.PIPE)
    print("✅ uv is already installed")
except subprocess.CalledProcessError:
    print("⬇️ uv not found. Installing uv...")
    subprocess.run(["pip", "install", "uv"], check=True)
    print("✅ uv installed successfully")

# 6. 📂 Enter repo directory
print(f"\n📂 [STEP 7] Switching to repo directory: {GITHUB_REPO_NAME}")
os.chdir(GITHUB_REPO_NAME)
print(f"📍 Current directory: {os.getcwd()}")

# 7. 🧾 Sync dependencies
print("\n🧾 [STEP 8] Syncing dependencies with uv...")
try:
    subprocess.run(["uv", "sync"], check=True)
    print("✅ All dependencies installed & synced successfully")
except subprocess.CalledProcessError:
    print("⚠️ uv sync failed — make sure pyproject.toml or requirements.txt exists")

print("\n🚀 Setup completed successfully! Your Colab environment is ready.")


def git_push(message="Auto commit from Colab"):
    print(f"📤 Pushing with commit message: {message}")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)
    print("✅ Changes pushed successfully!")

# Example usage:
# git_push("Updated training notebook")