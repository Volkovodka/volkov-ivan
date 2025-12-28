import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


# Функция для создания признаков
def create_features(df, is_train=True):
    df = df.copy()
    
    # Преобразование даты
    df['dt'] = pd.to_datetime(df['dt'])
    df['year'] = df['dt'].dt.year
    df['quarter'] = df['dt'].dt.quarter
    df['week'] = df['dt'].dt.isocalendar().week
    df['day'] = df['dt'].dt.day
    df['dayofweek'] = df['dt'].dt.dayofweek
    df['dayofyear'] = df['dt'].dt.dayofyear
    df['is_month_start'] = df['dt'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['dt'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['dt'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['dt'].dt.is_quarter_end.astype(int)
    
    # Взаимодействие категориальных признаков
    df['cat_interaction_1'] = df['management_group_id'].astype(str) + '_' + df['first_category_id'].astype(str)
    df['cat_interaction_2'] = df['second_category_id'].astype(str) + '_' + df['third_category_id'].astype(str)
    
    # Сезонные признаки
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    
    # Признаки на основе погоды
    df['temp_humidity_interaction'] = df['avg_temperature'] * df['avg_humidity']
    df['wind_precipitation_interaction'] = df['avg_wind_level'] * df['precpt']
    df['weather_severity'] = np.abs(df['avg_temperature']) + np.abs(df['avg_humidity']) + np.abs(df['avg_wind_level'])
    
    # Статистики по ценам
    if 'price_p05' in df.columns and 'price_p95' in df.columns:
        df['price_range'] = df['price_p95'] - df['price_p05']
        df['price_mid'] = (df['price_p05'] + df['price_p95']) / 2
    
    return df

# Создаем признаки для train
print("Создание признаков для train...")
train_features = create_features(train_df, is_train=True)

# Создаем признаки для test
print("Создание признаков для test...")
test_features = create_features(test_df, is_train=False)

# Кластеризация товаров для создания новых признаков
print("Кластеризация товаров...")
product_features = train_features.groupby('product_id').agg({
    'price_p05': ['mean', 'std'],
    'price_p95': ['mean', 'std'],
    'n_stores': ['mean', 'std'],
    'management_group_id': 'first',
    'first_category_id': 'first'
}).fillna(0)

product_features.columns = [f'product_{a}_{b}' if b != 'first' else f'product_{a}' 
                           for a, b in product_features.columns]

# Стандартизация для кластеризации
scaler_pca = StandardScaler()
product_scaled = scaler_pca.fit_transform(product_features.select_dtypes(include=[np.number]))

# Применяем PCA для снижения размерности
print("Применение PCA...")
pca = PCA(n_components=min(5, product_scaled.shape[1]), random_state=322)
product_pca = pca.fit_transform(product_scaled)

# Кластеризация на основе PCA компонент
kmeans = KMeans(n_clusters=min(8, len(product_pca)), random_state=322, n_init=10)
product_clusters = kmeans.fit_predict(product_pca)
product_features['product_cluster'] = product_clusters

# Добавляем кластеры к основным данным
product_cluster_map = product_features['product_cluster'].to_dict()
train_features['product_cluster'] = train_features['product_id'].map(product_cluster_map).fillna(-1).astype(int)
test_features['product_cluster'] = test_features['product_id'].map(product_cluster_map).fillna(-1).astype(int)

# Подготовка данных для обучения
feature_columns = [
    'n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
    'holiday_flag', 'activity_flag', 'management_group_id', 'first_category_id',
    'second_category_id', 'third_category_id', 'dow', 'day_of_month', 
    'week_of_year', 'month', 'year', 'quarter', 'week', 'day', 'dayofweek',
    'dayofyear', 'is_month_start', 'is_month_end', 'is_quarter_start', 
    'is_quarter_end', 'sin_month', 'cos_month', 'sin_day', 'cos_day',
    'temp_humidity_interaction', 'wind_precipitation_interaction', 'weather_severity',
    'product_cluster', 'cat_interaction_1', 'cat_interaction_2'
]

# Преобразуем категориальные взаимодействия в числовые
for df in [train_features, test_features]:
    df['cat_interaction_1'] = df['cat_interaction_1'].astype('category').cat.codes
    df['cat_interaction_2'] = df['cat_interaction_2'].astype('category').cat.codes

# Удаляем строки с пропусками в целевых переменных
train_features = train_features.dropna(subset=['price_p05', 'price_p95'])

X = train_features[feature_columns].fillna(0)
y = train_features[['price_p05', 'price_p95']]

# Детекция аномалий с помощью Isolation Forest
print("Детекция аномалий...")
iso_forest = IsolationForest(contamination=0.05, random_state=322)
outliers = iso_forest.fit_predict(X)
X_clean = X[outliers == 1]
y_clean = y.loc[X_clean.index]

print(f"Удалено аномалий: {len(X) - len(X_clean)}")

# Обучение модели
print("Обучение модели...")
# Используем RobustScaler для устойчивости к выбросам
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X_clean)

# Градиентный бустинг с мульти-таргет регрессией
model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=322,
        subsample=0.8
    )
)

model.fit(X_scaled, y_clean)
print("Модель обучена!")

# Подготовка тестовых данных
print("Подготовка тестовых данных...")
X_test = test_features[feature_columns].fillna(0)
X_test_scaled = robust_scaler.transform(X_test)

# Предсказание для теста
print("Предсказание для теста...")
predictions = model.predict(X_test_scaled)

# Создание сабмишена
submission = test_df.copy()
submission = submission[['row_id']]
submission['price_p05'] = predictions[:, 0]
submission['price_p95'] = predictions[:, 1]

# Пост-обработка: убедимся, что p05 <= p95
submission['price_p05'] = np.minimum(submission['price_p05'], submission['price_p95'] - 0.001)
submission['price_p95'] = np.maximum(submission['price_p95'], submission['price_p05'] + 0.001)

# Сохранение результатов
submission.to_csv('results/submission.csv', index=False)
print("Сабмишн сохранен в submission.csv")

# Валидация модели
print("\nВалидация модели:")
print("=" * 50)

# Рассчитаем IoU на временном сплите
from sklearn.model_selection import train_test_split

# Разделим данные на train/val с учетом времени
train_features_sorted = train_features.sort_values('dt')
split_idx = int(len(train_features_sorted) * 0.8)
X_train = train_features_sorted.iloc[:split_idx][feature_columns].fillna(0)
y_train = train_features_sorted.iloc[:split_idx][['price_p05', 'price_p95']]
X_val = train_features_sorted.iloc[split_idx:][feature_columns].fillna(0)
y_val = train_features_sorted.iloc[split_idx:][['price_p05', 'price_p95']]

# Обучим модель на train
robust_scaler_val = RobustScaler()
X_train_scaled = robust_scaler_val.fit_transform(X_train)
model_val = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=322
    )
)
model_val.fit(X_train_scaled, y_train)

# Предсказание на validation
X_val_scaled = robust_scaler_val.transform(X_val)
val_preds = model_val.predict(X_val_scaled)

# Функция для расчета IoU
def calculate_iou(true_p05, true_p95, pred_p05, pred_p95, epsilon=1e-6):
    # "Утолщение" интервалов
    true_p05 = true_p05 - epsilon
    true_p95 = true_p95 + epsilon
    pred_p05 = pred_p05 - epsilon
    pred_p95 = pred_p95 + epsilon
    
    intersection = np.maximum(0, np.minimum(true_p95, pred_p95) - np.maximum(true_p05, pred_p05))
    union = (true_p95 - true_p05) + (pred_p95 - pred_p05) - intersection
    iou = intersection / (union + 1e-10)
    return np.mean(iou)

iou_score = calculate_iou(y_val['price_p05'].values, y_val['price_p95'].values, 
                          val_preds[:, 0], val_preds[:, 1])
print(f"IoU score на валидации: {iou_score:.4f}")

# Статистики предсказаний
print("\nСтатистики предсказаний:")
print(f"Min price_p05: {submission['price_p05'].min():.4f}")
print(f"Max price_p95: {submission['price_p95'].max():.4f}")
print(f"Mean price range: {(submission['price_p95'] - submission['price_p05']).mean():.4f}")
print(f"Std price range: {(submission['price_p95'] - submission['price_p05']).std():.4f}")

# Анализ важности признаков
print("\nАнализ важности признаков (первые 10):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.estimators_[0].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(10))

print("\nГотово! Проверьте файл submission.csv")
