#!/bin/bash

# ============================
# 🚀 COLAB GIT SESSION STARTER
# ============================

# 1. Load environment variables from .env file
set -a
source .env
set +a

# 2. ✅ Configure Git
git config --global user.email "$GITHUB_EMAIL"
git config --global user.name "$GITHUB_USERNAME"

# 3. 🔑 Setup GitHub Personal Access Token (PAT) securely
USERNAME="$GITHUB_USERNAME"
TOKEN="$GITHUB_PAT"

# Create a .netrc file so PAT doesn't appear in URLs
echo "machine github.com login $USERNAME password $TOKEN" > ~/.netrc
chmod 600 ~/.netrc

# 4. 🧰 Repo details
REPO_URL="$GITHUB_REPO_URL"
REPO_NAME="$GITHUB_REPO_NAME"   # ✅ Added missing quote

# 5. 🌀 Clone if not already present, else pull
if [ ! -d "$REPO_NAME" ]; then
    git clone "$REPO_URL" "$REPO_NAME"
    cd "$REPO_NAME" || exit
else
    cd "$REPO_NAME" || exit
    git pull
fi

echo "✅ Git setup completed with .env file"


# Quick push helper function
if [ "$1" = "push" ]; then
    git add .
    git commit -m "${2:-Auto commit from Colab}"
    git push origin main
fi

# Example usage goes like - 
# bash: colab_setup.sh push "your commit message"