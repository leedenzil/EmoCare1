# Text Model Improvements to Reach 90% Accuracy

## Immediate Fixes (Expected: 64% → 72-75%)

1. **Fix Loss Function Bug** (same bug as video model)
   - Apply class weights to loss values, not target distribution
   
2. **Increase MAX_LENGTH to 512**
   - Capture full Reddit post context
   - Reddit posts often exceed 128 tokens

## Major Upgrades (Expected: 75% → 85-90%)

3. **Switch to Better Model**
   - Option A: `cardiffnlp/twitter-roberta-base-sentiment-latest` (social media optimized)
   - Option B: `microsoft/deberta-v3-base` (state-of-the-art)
   - Option C: `roberta-base` (strong baseline)

4. **More Training Epochs**
   - 15-20 epochs with early stopping (patience=5)
   
5. **Better Text Preprocessing**
   - Remove URLs
   - Handle emojis
   - Clean Reddit markdown

6. **Advanced Techniques**
   - Focal loss for hard examples
   - Learning rate warmup + cosine decay
   - Gradient accumulation for larger effective batch size
   - Test-time augmentation (back-translation)

## Recommended Configuration for 90%

```python
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
MAX_LENGTH = 512  # Full context
BATCH_SIZE = 16   # Larger model needs smaller batch
EPOCHS = 20
LEARNING_RATE = 1e-5  # Lower for pre-trained sentiment model
WARMUP_RATIO = 0.1
```
