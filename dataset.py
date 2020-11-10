import collections
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

_THUCNews = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"
def load_THUCNews_title_label(file=_THUCNews):
    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    titles = []
    labels = []
    for line in lines:
        title, label = line.split("\t")
        if not title:
            continue
        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label:i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

class CharTokenizer:
    """字转ID
    """

    def __init__(self, min_freq=16):
        self.char2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.min_freq = min_freq

    def fit(self, X):
        # 建立词ID映射表
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1
        # 过滤低频词
        chars = {i:j for i, j in chars.items() if j >= self.min_freq}
        # 0:MASK
        # 1:UNK
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNKNOW))
            ids.append(s)
        return ids

    def fit_transform(self, X):
        self.fit(X)
        ids = self.transform(X)
        return ids

    def __len__(self):
    	return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

    @property
    def vocab(self):
        return self.char2id

# 处理数据
X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=6732)

num_classes = len(classes)
# 转化成字id
tokenizer = CharTokenizer()
tokenizer.fit(X_train)
X_train = tokenizer.transform(X_train)
X_test = tokenizer.transform(X_test)

maxlen = 48

X_train = sequence.pad_sequences(
    X_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0.0)

X_test = sequence.pad_sequences(
    X_test, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0.0)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

num_chars = len(tokenizer)
