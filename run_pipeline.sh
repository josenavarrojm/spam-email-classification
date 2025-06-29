#!/bin/bash

echo "🚀 Running Spam Email Classification pipeline..."

# Step 1: Data preparation
echo "🔧 Step 1: Preparing data..."
python src/data_prep.py

# Step 2: Train models
echo "🧠 Step 2: Training models..."
python src/train_model.py

# Step 3: Evaluate models
echo "📊 Step 3: Evaluating models..."
python src/evaluate.py

# Step 4: Done!
echo "✅ Pipeline completed successfully!"
