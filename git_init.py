#!/usr/bin/env python3
"""Git initialization and push script for AI-Reader project."""

import subprocess
import os
import sys

def run_command(cmd, description, check=True):
    """Execute a command and return output."""
    print(f"\n[*] {description}")
    print(f"    Command: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=r"E:\Research\RAG",
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"    Output: {result.stdout.strip()}")
        if result.returncode != 0 and check:
            print(f"    ERROR: {result.stderr.strip()}")
            return False
        if result.stderr and check:
            print(f"    STDERR: {result.stderr.strip()}")
        return True, result.stdout, result.stderr
    except Exception as e:
        print(f"    EXCEPTION: {e}")
        return False, "", str(e)

def main():
    os.chdir(r"E:\Research\RAG")
    
    print("=" * 60)
    print("Git Repository Initialization for AI-Reader")
    print("=" * 60)
    
    # Step 1: Check if git repo exists
    print("\n[Step 1] Checking if git repository is initialized...")
    if os.path.isdir(".git"):
        print("    ✓ Git repository already initialized")
    else:
        print("    → Initializing git repository...")
        success, stdout, stderr = run_command("git init", "Initializing git repo")
        if not success:
            print("    ✗ Failed to initialize git")
            sys.exit(1)
        print("    ✓ Git repository initialized")
    
    # Step 2: Rename branch to main
    print("\n[Step 2] Ensuring default branch is 'main'...")
    success, current_branch, _ = run_command(
        'git branch --show-current',
        "Checking current branch",
        check=False
    )
    current_branch = current_branch.strip() if current_branch else "master"
    print(f"    Current branch: {current_branch}")
    
    if current_branch != "main":
        print(f"    → Renaming branch from '{current_branch}' to 'main'...")
        success, _, _ = run_command("git branch -M main", "Renaming branch to main")
        if not success:
            print("    ✗ Failed to rename branch")
            sys.exit(1)
        print("    ✓ Branch renamed to 'main'")
    else:
        print("    ✓ Already on 'main' branch")
    
    # Step 3: Configure remote origin
    print("\n[Step 3] Configuring remote origin...")
    target_url = "git@github.com:Wesirehour/AI-Reader.git"
    success, existing_url, _ = run_command(
        'git config --get remote.origin.url',
        "Checking existing remote URL",
        check=False
    )
    existing_url = existing_url.strip() if existing_url else None
    
    if existing_url:
        print(f"    Existing remote URL: {existing_url}")
        if existing_url != target_url:
            print(f"    → Updating remote URL to {target_url}...")
            success, _, _ = run_command(
                f'git remote set-url origin {target_url}',
                "Updating remote URL"
            )
            if not success:
                print("    ✗ Failed to update remote URL")
                sys.exit(1)
            print("    ✓ Remote URL updated")
        else:
            print("    ✓ Remote URL already correct")
    else:
        print(f"    → Adding remote origin: {target_url}...")
        success, _, _ = run_command(
            f'git remote add origin {target_url}',
            "Adding remote origin"
        )
        if not success:
            print("    ✗ Failed to add remote origin")
            sys.exit(1)
        print("    ✓ Remote origin added")
    
    # Step 4: Stage all files
    print("\n[Step 4] Staging files...")
    success, _, _ = run_command("git add -A", "Staging all files")
    if not success:
        print("    ✗ Failed to stage files")
        sys.exit(1)
    print("    ✓ Files staged")
    
    # Step 5: Create commit
    print("\n[Step 5] Creating commit...")
    commit_cmd = (
        'git commit -m "chore: initialize AI-Reader project" '
        '-m "" '
        '-m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"'
    )
    success, stdout, stderr = run_command(commit_cmd, "Creating commit", check=False)
    
    # Check if commit succeeded or if files are already committed
    if success and "create mode" in stdout:
        print("    ✓ Commit created successfully")
    elif "nothing to commit" in stderr or "nothing to commit" in stdout:
        print("    ℹ Nothing to commit (repository already initialized)")
    elif success:
        print("    ✓ Commit created")
    else:
        print(f"    ⚠ Commit status unclear: {stderr}")
    
    # Step 6: Get commit hash
    print("\n[Step 6] Getting commit information...")
    success, commit_hash, _ = run_command(
        "git rev-parse HEAD",
        "Getting commit hash"
    )
    if success:
        commit_hash = commit_hash.strip()
        print(f"    ✓ Latest commit: {commit_hash[:8]}")
    else:
        print("    ✗ Could not get commit hash")
        sys.exit(1)
    
    # Step 7: Verify remote
    print("\n[Step 7] Verifying remote configuration...")
    success, remote_output, _ = run_command(
        "git remote -v",
        "Getting remote configuration"
    )
    if success:
        print(f"    ✓ Remote configuration:\n{remote_output}")
    
    # Step 8: Push to GitHub
    print("\n[Step 8] Pushing to GitHub...")
    print("    → Running: git push -u origin main")
    push_result = subprocess.run(
        "git push -u origin main",
        shell=True,
        cwd=r"E:\Research\RAG",
        capture_output=True,
        text=True
    )
    
    if push_result.returncode == 0:
        print("    ✓ Push successful!")
        print(f"    Output: {push_result.stdout.strip()}")
    else:
        print("    ✗ Push failed!")
        print(f"    Error: {push_result.stderr.strip()}")
        print("\n[!] AUTHENTICATION/SSH ERROR DETECTED")
        print("    The push failed. This is likely due to:")
        print("    - Missing SSH key setup")
        print("    - SSH key not added to ssh-agent")
        print("    - GitHub not recognizing the SSH key")
        print("\n    No destructive changes were made. Repository is ready to push when SSH is configured.")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Current branch: main")
    print(f"✓ Remote URL: {target_url}")
    print(f"✓ Latest commit: {commit_hash[:8]}")
    print(f"✓ Status: Successfully pushed to GitHub")
    print("=" * 60)

if __name__ == "__main__":
    main()
