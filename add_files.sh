#!/bin/bash
for file in ADVANCED.md CLAUDE_INSTALL_GUIDE.md INSTALL.md install.sh install_tektra.py run.py start.py test_installation.py; do
  echo "Adding $file"
  git add "$file" && echo "Success: $file" || echo "Failed: $file"
done
