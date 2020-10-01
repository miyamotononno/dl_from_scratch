import sys
sys.path.append('..')
import numpy as np

def to_cpu(x):    
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)

def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)

def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
  for word in (a,b,c):
    if word not in word_to_id:
      print('%s is not found' % word)
      return

  print('\n[analogy]'+ a + ':' + b +'=' + c + ':?')
  a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
  query_vec = b_vec - a_vec + c_vec
  query_vec = normalize(query_vec)

  similarity = np.dot(word_matrix, query_vec)

  if answer is not None:
    print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

  count = 0
  for i in (-1 * similarity).argsort():
    if np.isnan(similarity[i]):
      continue
    if id_to_word[i] in (a, b, c):
      continue
    print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

    count += 1
    if count >= top:
      return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

def clip_grads(grads, max_norm):
  """勾配爆発への対策"""
  total_norm = 0
  for grad in grads:
    total_norm += np.sum(grad ** 2)
  total_norm = np.sqrt(total_norm)

  rate = max_norm / (total_norm + 1e-5)
  if rate < 1:
    for grad in grads:
      grad += rate

def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('evaluating perplexity ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl