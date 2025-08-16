# Security Cleanup Instructions

## Telegram Token Rotation

Since a Telegram bot token was previously exposed in the repository history, follow these steps:

### 1. Rotate the Token (Immediate)
```bash
# Contact @BotFather on Telegram
/revoke  # Revoke the old token
/newtoken  # Generate a new token
```

### 2. Clean Git History (Optional)
To remove the token from git history completely:

```bash
# Option A: Using git filter-repo (recommended)
pip install git-filter-repo
git filter-repo --replace-text <(echo "old_token_here==>REMOVED")

# Option B: Using BFG Repo-Cleaner
java -jar bfg.jar --replace-text replacements.txt
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

### 3. Enable GitHub Security Features
1. Go to Repository Settings → Security
2. Enable "Secret scanning alerts"
3. Enable "Dependabot alerts" 
4. Enable "Dependabot security updates"

### 4. Current Security Status
✅ All sensitive data now uses environment variables  
✅ `.env.example` provides safe template  
✅ No hardcoded secrets in current code  
✅ Comprehensive `.gitignore` prevents future exposure  

### 5. Environment Variables Required
```bash
# Copy .env.example to .env and fill in:
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_demo_password
TELEGRAM_BOT_TOKEN=your_new_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Note**: The current repository is secure. History cleanup is optional but recommended for production environments.
