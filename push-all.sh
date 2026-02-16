#!/bin/bash
# Push to all remotes with correct method for each target

set -e

echo "=== Pushing to GitHub (full repo) ==="
git push github main

echo ""
echo "=== Pushing to HF exp-7 (subtree: experiments/exp-7/) ==="
git subtree push --prefix=experiments/exp-7 origin main

echo ""
echo "=== Pushing to HF exp-8 (subtree: experiments/exp-8/) ==="
git subtree push --prefix=experiments/exp-8 vad main

echo ""
echo "=== All pushes complete ==="
