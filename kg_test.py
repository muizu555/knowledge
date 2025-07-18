import pykeen.datasets
from pykeen.pipeline import pipeline

# 1. データセットを指定 (WN18)
dataset = pykeen.datasets.WN18()

# 2. パイプライン実行 (TransEで学習)
result = pipeline(
    training=dataset.training,
    testing=dataset.testing,
    model='TransE',
    epochs=5,  # 短くして簡単に
)

# 3. リンク予測の評価結果を表示
print(result.get_metric('mean_rank'))
print(result.get_metric('hits@10'))
