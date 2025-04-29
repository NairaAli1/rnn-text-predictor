# rnn-text-predictor
# rnn-text-predictor
import numpy as np

word_to_idx = {"I": 0, "love": 1, "deep": 2, "learning": 3}
idx_to_word = {v: k for k, v in word_to_idx.items()}

input_seq = [0, 1, 2]
target = 3
def one_hot(index, vocab_size):
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec

vocab_size = len(word_to_idx)
inputs = [one_hot(i, vocab_size) for i in input_seq]
target_vec = one_hot(target, vocab_size)

# إعداد الأوزان عشوائيًا
hidden_size = 5
Wxh = np.random.randn(hidden_size, vocab_size)   
Whh = np.random.randn(hidden_size, hidden_size)  
Why = np.random.randn(vocab_size, hidden_size)   
bh = np.zeros((hidden_size, 1))               
by = np.zeros((vocab_size, 1))                 

# التدريب
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

learning_rate = 0.1
h_prev = np.zeros((hidden_size, 1))

for epoch in range(1000):
    hs = {}
    hs[-1] = h_prev
    loss = 0
    
    # Forward pass
    for t in range(len(inputs)):
        x = inputs[t].reshape(-1, 1)
        hs[t] = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs[t-1]) + bh)
        
    y = np.dot(Why, hs[len(inputs) - 1]) + by
    p = softmax(y)
    
    loss = -np.sum(target_vec.reshape(-1, 1) * np.log(p))
    dWhy = np.dot((p - target_vec.reshape(-1, 1)), hs[len(inputs) - 1].T)
    dby = p - target_vec.reshape(-1, 1)
    Why -= learning_rate * dWhy
    by -= learning_rate * dby

    if epoch % 100 == 0:
        pred_idx = np.argmax(p)
        print(f"Epoch {epoch}: Prediction = {idx_to_word[pred_idx]}, Loss = {loss:.4f}")
print("\nFinal prediction after training:")
pred_idx = np.argmax(p)
print(f"Predicted 4th word: {idx_to_word[pred_idx]}")
