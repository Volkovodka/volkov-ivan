import pandas as pd
import numpy as np
import re
import gc
import warnings
import random
import os
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

SEED = 993
np.random.seed(SEED)
random.seed(SEED)


def preprocess_text(text):
    """Нормализация текста: нижний регистр, только слова/пробелы/дефисы"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s\-+]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_features(df):
    """Создание расширенного набора признаков"""
    features = pd.DataFrame(index=df.index)

    def compute_text_overlap(row):
        q_words = set(row['query_clean'].split())
        t_words = set(row['title_clean'].split())
        if not q_words or not t_words:
            return 0.0, 0.0, 0.0
        intersection = q_words & t_words
        union = q_words | t_words
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap = len(intersection) / len(q_words) if q_words else 0.0
        coverage = sum(1 for w in q_words if w in t_words) / len(q_words) if q_words else 0.0
        return jaccard, overlap, coverage

    results = df.apply(compute_text_overlap, axis=1, result_type='expand')
    features['jaccard'] = results[0]
    features['word_overlap'] = results[1]
    features['coverage'] = results[2]

    features['exact_match'] = df.apply(
        lambda x: 1.0 if x['query_clean'] in x['title_clean'] else 0.0, axis=1
    )
    features['starts_with_query'] = df.apply(
        lambda x: 1.0 if x['title_clean'].startswith(x['query_clean'][:15]) else 0.0, axis=1
    )

    features['query_len'] = df['query_clean'].str.len()
    features['title_len'] = df['title_clean'].str.len()
    features['desc_len'] = df['desc_clean'].str.len()

    features['query_words'] = df['query_clean'].apply(lambda x: len(x.split()))
    features['title_words'] = df['title_clean'].apply(lambda x: len(x.split()))

    features['len_ratio'] = features['title_len'] / (features['query_len'] + 1)
    features['word_ratio'] = features['title_words'] / (features['query_words'] + 1)

    features['digits_in_title'] = df['title_clean'].apply(lambda x: 1.0 if any(c.isdigit() for c in x) else 0.0)
    features['digits_in_query'] = df['query_clean'].apply(lambda x: 1.0 if any(c.isdigit() for c in x) else 0.0)

    # Категориальные признаки
    for col in ['product_brand', 'product_color', 'product_locale']:
        if col in df.columns:
            le = LabelEncoder()
            features[f'{col}_enc'] = le.fit_transform(df[col].astype(str))

    features['query_pos_in_title'] = df.apply(
        lambda x: x['title_clean'].find(x['query_clean'][:10]) / max(len(x['title_clean']), 1), axis=1
    )

    features['composite_score'] = (
        features['exact_match'] * 0.5 +
        features['word_overlap'] * 0.3 +
        features['coverage'] * 0.2
    )

    return features


def create_submission(predictions, test_df):
    """Создаёт submission файл в формате: id,prediction"""
    test_df = test_df.copy()
    if 'id' not in test_df.columns:
        test_df['id'] = range(len(test_df))
    
    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'prediction': predictions
    }).drop_duplicates(subset=['id']).reset_index(drop=True)

    os.makedirs('results', exist_ok=True)
    path = 'results/submission.csv'
    submission.to_csv(path, index=False)
    print(f"✓ Submission сохранён: {path}")
    return path


def main():
    print("=" * 70)
    print("РЕШЕНИЕ ДЛЯ СОРТИРОВКИ ПРОДУКТОВ ПО ЗАПРОСАМ")
    print("=" * 70)

    # === ШАГ 1: ЗАГРУЗКА ДАННЫХ ===
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"\n[1/8] Данные загружены: train={train_df.shape}, test={test_df.shape}")

    # === ШАГ 2: ТЕКСТОВАЯ ПОДГОТОВКА ===
    for df in [train_df, test_df]:
        df['query_clean'] = df['query'].fillna('').apply(preprocess_text)
        df['title_clean'] = df['product_title'].fillna('').apply(preprocess_text)
        df['desc_clean'] = df['product_description'].fillna('').apply(preprocess_text)
        for col in ['product_brand', 'product_color', 'product_locale']:
            if col in df.columns:
                df[col] = df[col].fillna('missing').astype(str)
    print("[2/8] Текст обработан")

    # === ШАГ 3: БАЗОВЫЕ ПРИЗНАКИ ===
    X_train = create_features(train_df)
    X_test = create_features(test_df)
    print("[3/8] Базовые признаки созданы")

    # === ШАГ 4: TF-IDF + SVD ===
    all_texts = pd.concat([
        train_df['query_clean'] + ' _sep_ ' + train_df['title_clean'],
        test_df['query_clean'] + ' _sep_ ' + test_df['title_clean']
    ])
    vectorizer = TfidfVectorizer(max_features=150, ngram_range=(1, 3), min_df=2, max_df=0.9, dtype=np.float32)
    vectorizer.fit(all_texts)

    tfidf_train = vectorizer.transform(train_df['query_clean'] + ' _sep_ ' + train_df['title_clean'])
    tfidf_test = vectorizer.transform(test_df['query_clean'] + ' _sep_ ' + test_df['title_clean'])

    svd = TruncatedSVD(n_components=25, random_state=SEED)
    svd_train = svd.fit_transform(tfidf_train)
    svd_test = svd.transform(tfidf_test)

    for i in range(svd_train.shape[1]):
        X_train[f'semantic_{i}'] = svd_train[:, i]
        X_test[f'semantic_{i}'] = svd_test[:, i]
    print("[4/8] TF-IDF + SVD признаки добавлены")

    # === ШАГ 5: ГРУППОВЫЕ СТАТИСТИКИ ===
    train_group_sizes = train_df.groupby('query_id').size()
    test_group_sizes = test_df.groupby('query_id').size()
    X_train['group_size'] = train_df['query_id'].map(train_group_sizes)
    X_test['group_size'] = test_df['query_id'].map(test_group_sizes)

    numeric_cols = ['query_len', 'title_len', 'jaccard', 'word_overlap']
    for col in numeric_cols:
        if col in X_train.columns:
            group_means = train_df.groupby('query_id').apply(lambda x: X_train.loc[x.index, col].mean())
            X_train[f'group_mean_{col}'] = train_df['query_id'].map(group_means)
            X_test[f'group_mean_{col}'] = X_test[col].mean()
            X_train[f'group_dev_{col}'] = X_train[col] - X_train[f'group_mean_{col}']
            X_test[f'group_dev_{col}'] = X_test[col] - X_test[f'group_mean_{col}']
    print("[5/8] Групповые признаки добавлены")

    # === ШАГ 6: ФИНАЛЬНАЯ ПОДГОТОВКА ===
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Стандартизация некатегориальных признаков
    cat_cols = [col for col in X_train.columns if '_enc' in col]
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols, index=X_test.index
    )

    # Объединяем обратно с категориальными
    for col in cat_cols:
        X_train_scaled[col] = X_train[col].values
        X_test_scaled[col] = X_test[col].values

    X_train_final = X_train_scaled.astype('float32')
    X_test_final = X_test_scaled.astype('float32')

    y_train = train_df['relevance'].values
    groups = train_df['query_id'].values
    print(f"[6/8] Данные готовы: признаков={X_train_final.shape[1]}")

    # === ШАГ 7: ОБУЧЕНИЕ МОДЕЛИ ===
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 8,
        'learning_rate': 0.05,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.3,
        'random_state': SEED,
        'n_estimators': 500,
        'verbosity': -1,
        'n_jobs': -1,
    }

    gkf = GroupKFold(n_splits=5)
    test_preds = np.zeros(len(X_test_final))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_final, y_train, groups), 1):
        print(f"\n[7/8] Обучение фолда {fold}/{gkf.n_splits}")

        X_tr, y_tr = X_train_final.iloc[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train_final.iloc[val_idx], y_train[val_idx]
        g_tr, g_val = groups[tr_idx], groups[val_idx]

        train_groups = [np.sum(g_tr == gid) for gid in np.unique(g_tr)]
        val_groups = [np.sum(g_val == gid) for gid in np.unique(g_val)]

        train_ds = lgb.Dataset(X_tr, label=y_tr, group=train_groups, free_raw_data=False)
        val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds, free_raw_data=False)

        model = lgb.train(
            params,
            train_ds,
            valid_sets=[val_ds],
            num_boost_round=params['n_estimators'],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        test_preds += model.predict(X_test_final) / gkf.n_splits
        del model, train_ds, val_ds
        gc.collect()

    # === ШАГ 8: ПОСТОБРАБОТКА ===
    final_preds = test_preds.copy()

    print("\n[8/8] Постобработка: усиление топа...")
    for query_id in tqdm(test_df['query_id'].unique()):
        mask = test_df['query_id'] == query_id
        idxs = np.where(mask)[0]
        if len(idxs) <= 1:
            continue
        scores = final_preds[mask]
        if scores.std() > 0:
            scores = (scores - scores.mean()) / scores.std()
        exp_scores = np.exp(scores - np.max(scores))
        softmax = exp_scores / exp_scores.sum()
        final_preds[mask] = 0.6 * scores + 0.4 * softmax

        top_k = min(3, len(scores))
        top_idx = np.argsort(final_preds[mask])[-top_k:]
        for i, idx in enumerate(top_idx):
            boost = 1.0 + (i + 1) * 0.05
            orig_idx = idxs[idx]
            final_preds[orig_idx] *= boost

    print("Постобработка: ранжирование...")
    for query_id in tqdm(test_df['query_id'].unique()):
        mask = test_df['query_id'] == query_id
        group_scores = final_preds[mask]
        if len(group_scores) > 1:
            ranks = stats.rankdata(group_scores, method='ordinal')
            max_rank = len(group_scores)
            norm_ranks = (max_rank - ranks + 1) / max_rank
            final_preds[mask] = 0.5 * group_scores + 0.5 * norm_ranks

    # Нормализация по всему тесту
    final_preds = (final_preds - final_preds.mean()) / (final_preds.std() + 1e-8)

    # Сохранение
    create_submission(final_preds, test_df)

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    main()